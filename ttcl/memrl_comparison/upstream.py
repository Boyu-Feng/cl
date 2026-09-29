"""Execute unmodified upstream definitions without importing the MemOS service.

Only imports/initialization are replaced. Every selected function/class AST is
compiled as supplied by upstream, including native retrieval and Q updates.
This is a method adaptation, not the complete upstream MemOS deployment.
"""
from __future__ import annotations
import ast
import hashlib
import logging
from pathlib import Path
import sys
import types
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field


class TextualMemoryMetadata(BaseModel):
    model_config = ConfigDict(extra='allow')


class TextualMemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    memory: str
    metadata: TextualMemoryMetadata


SOURCES = ('providers/base.py', 'providers/llm.py', 'service/strategies.py', 'service/builders.py',
           'service/updater.py', 'service/value_driven.py', 'utils/task_id.py',
           'service/memory_service.py')


def load(root):
    root = Path(root)
    namespace = {'TextualMemoryMetadata': TextualMemoryMetadata,
                 'TextualMemoryItem': TextualMemoryItem}
    hashes = {}
    for relative in SOURCES:
        path = root/'memrl'/relative
        source = path.read_bytes()
        tree = ast.parse(source)
        if relative=='providers/llm.py':
            provider=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OpenAILLM')
            # Preserve the actual provider's script prompt override, not merely
            # the generic BaseLLM fallback. HTTP/API code is our local transport.
            tree=ast.Module(body=[next(n for n in provider.body if isinstance(n,ast.FunctionDef)
                                      and n.name=='generate_script')],type_ignores=[])
        # Runtime imports inside upstream method bodies are retained. We never
        # call the MemOS-specific constructor, server, or snapshot methods.
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
                if not getattr(node, 'level', 0) and all(
                        n.split('.')[0] not in {'memos','memrl'} for n in names):
                    if '__future__' not in names:
                        nodes.append(node)
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module_ast = ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[]))
        name = '_ttcl_memrl_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        module = types.ModuleType(name)
        module.__dict__.update(namespace, logger=logging.getLogger(name))
        module.__dict__['__name__'] = name
        sys.modules[name] = module
        exec(compile(module_ast, str(path), 'exec'), module.__dict__)
        namespace.update({k:v for k,v in module.__dict__.items() if not k.startswith('__')})
        hashes[relative] = {'file_sha256': hashlib.sha256(source).hexdigest(),
            'executed_ast_sha256': hashlib.sha256(ast.dump(module_ast).encode()).hexdigest(),
            'definitions': [n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))]}
    return types.SimpleNamespace(**namespace), hashes
