"""Compatibility command for the sealed-demo FLASH preparation adapter.

The GUI and existing scripts still invoke this path. Preparation itself lives
with the FLASH policy in ``lerobot.policies.flash.sealed_demo_import``.
"""

from lerobot.policies.flash.sealed_demo_import import main

if __name__ == "__main__":
    main()
