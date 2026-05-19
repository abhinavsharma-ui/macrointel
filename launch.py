from project.run import MacroIntelligenceSystem

print("\n[+] Initializing Safe Custom Launcher...")

# 1. Save the original method that fetches your system's real symbols
original_get_symbols = MacroIntelligenceSystem._get_target_symbols

# 2. Create a wrapper that intercepts the list and slices it
def safe_get_symbols(self, market="full"):
    # Fetch your actual native symbols
    symbols = original_get_symbols(self, market=market)

    # If the list is huge, slice it safely to 150
    if isinstance(symbols, list) and len(symbols) > 150:
        print(f"\n[!!!] PROTECTING MEMORY: Slicing native {len(symbols)} symbol list down to 150 [!!!]\n")
        return symbols[:150]
    return symbols

# 3. Apply the patch to the engine in memory
MacroIntelligenceSystem._get_target_symbols = safe_get_symbols

# 4. Boot the engine
if __name__ == "__main__":
    try:
        print("[+] Booting MacroIntelligenceSystem...")
        engine = MacroIntelligenceSystem()
        engine.run()
    except KeyboardInterrupt:
        print("\n[!] Shutdown by user.")
    except Exception as e:
        print(f"\n[!] Engine crashed: {e}")
        import traceback
        traceback.print_exc()
