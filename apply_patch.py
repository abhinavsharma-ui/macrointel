import re

path = 'project/run.py'
with open(path, 'r') as f:
    code = f.read()

patch = """if __name__ == "__main__":
    print("\\n[!] INJECTING MEMORY SNIPER PATCH...\\n")
    original_method = MacroIntelligenceSystem._get_target_symbols
    def capped_method(self, market="full"):
        syms = original_method(self, market=market)
        if isinstance(syms, list) and len(syms) > 150:
            print(f"\\n[!] INTERCEPTED: Slicing {len(syms)} symbols down to 150 for ML stability.\\n")
            return syms[:150]
        return syms
    MacroIntelligenceSystem._get_target_symbols = capped_method
"""

if "capped_method" not in code:
    new_code = re.sub(r'^if __name__ == [\'"]__main__[\'"]:', patch, code, flags=re.MULTILINE)
    with open(path, 'w') as f:
        f.write(new_code)
    print("Patch successfully applied!")
else:
    print("Patch was already applied.")
