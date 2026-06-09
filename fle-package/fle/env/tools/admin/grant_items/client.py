from typing import Dict
from fle.env.tools import Tool


class GrantItems(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        items: Dict[str, int] = None,
    ):
        """
        Grant items to the player inventory, capped at a maximum.
        Used for periodic ammo top-ups in tower defense.
        :param items: Dict of item_name -> count to grant
        :return: Dict of items actually granted
        """
        if items is None:
            items = {
                "firearm-magazine": 20,
                "stone-wall": 10,
                "gun-turret": 2,
            }

        # Convert dict to Lua-friendly format
        items_str = ", ".join(
            f'["{k}"] = {v}' for k, v in items.items()
        )
        response, _ = self.execute(self.player_index, items_str)
        return self.clean_response(response)
