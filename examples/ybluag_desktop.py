"""Launch the native Yb:LuAG simulator from the repository root."""

from desktop_simulation import DesktopSimulation


if __name__ == "__main__":
    DesktopSimulation(initial_material="Yb:LuAG").mainloop()
