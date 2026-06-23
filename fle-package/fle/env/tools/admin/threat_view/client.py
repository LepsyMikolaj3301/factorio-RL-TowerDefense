from typing import Dict, List, Tuple

from fle.env.tools import Tool


class ThreatView(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        radius: float = 96.0,
        cell: float = 6.0,
        charted_only: bool = True,
        max_groups: int = 12,
        max_nests: int = 8,
    ) -> Dict[str, List[Tuple]]:
        """
        Cluster live enemy units into groups (swarms) and list nests (spawners).

        :return: {"groups": [(cx, cy, count, spread, vx, vy), ...],
                  "nests":  [(x, y, health), ...]}  — WORLD coordinates / raw
                 values. The env converts these to radar-relative normalized
                 observation features.
        """
        cmd = (
            "/silent-command rcon.print(storage.actions.threat_view("
            f"{self.player_index}, {float(center_x)}, {float(center_y)}, "
            f"{float(radius)}, {float(cell)}, {str(bool(charted_only)).lower()}, "
            f"{int(max_groups)}, {int(max_nests)}))"
        )
        resp = self.connection.rcon_client.send_command(cmd)
        groups: List[Tuple] = []
        nests: List[Tuple] = []
        if not isinstance(resp, str):
            return {"groups": groups, "nests": nests}

        g_section, _, n_section = resp.strip().partition("|")
        g_section = g_section[2:] if g_section.startswith("G:") else g_section
        n_section = n_section[2:] if n_section.startswith("N:") else n_section

        for rec in g_section.split(";"):
            if not rec:
                continue
            try:
                cx, cy, count, spread, vx, vy = rec.split(",")
                groups.append(
                    (float(cx), float(cy), int(float(count)),
                     float(spread), float(vx), float(vy))
                )
            except (ValueError, TypeError):
                continue

        for rec in n_section.split(";"):
            if not rec:
                continue
            try:
                x, y, hp = rec.split(",")
                nests.append((float(x), float(y), float(hp)))
            except (ValueError, TypeError):
                continue

        return {"groups": groups, "nests": nests}
