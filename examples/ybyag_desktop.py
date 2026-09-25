"""Launch the native Yb:YAG amplifier, with separate LuAG and Ho tabs."""
from desktop_simulation import DesktopSimulation

if __name__ == '__main__':
    DesktopSimulation(initial_material='Yb:YAG').mainloop()
