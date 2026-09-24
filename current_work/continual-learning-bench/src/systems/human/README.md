# Human

Interactive CLI baseline. A human plays the role of the system: the benchmark prints each query and any pending feedback to the terminal, and the user fills in the response fields one at a time. Here we have `supports_baseline = False` and `parallel_safe = False` — only one instance can run at a time because it reads from stdin.