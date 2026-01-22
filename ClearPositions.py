from paper_trader import load_state, save_state

def main():
    state = load_state()
    positions = state.get("positions", [])
    count = len(positions)
    state["positions"] = []
    # also clear last_processed_bar markers to avoid stale keys
    state["last_processed_bar"] = {}
    save_state(state)
    print(f"[Clear] Removed {count} open position(s) and reset last_processed markers.")

if __name__ == "__main__":
    main()
