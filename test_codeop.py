import codeop

def check_syntax(code_str):
    try:
        res = codeop.compile_command(code_str, symbol="exec")
        if res is None:
            return "Incomplete"
        return "Complete"
    except SyntaxError as e:
        return f"Invalid: {e}"

print("1:", check_syntax("def foo():\n"))
print("2:", check_syntax("def foo():\n    return 1\n"))
print("3:", check_syntax("def foo():\n    return return 1\n"))
print("4:", check_syntax("def foo(\n"))
print("5:", check_syntax("for i in range(10):\n"))
print("6:", check_syntax("for i in range(10]\n"))
