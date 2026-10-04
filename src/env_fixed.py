"""baseline MahjongGBEnv 的修复版,得分/裁决语义对齐官方裁判(judge/main.cpp)。

与 baseline 的差异(全部有 judge 源码与全量对拍证据,见 src/env_certify.py):
1. 自摸"和绝张":baseline 在 step() 的自摸分支先 `shownTiles[curTile] += 1`
   再查番,is4thTile 把自摸牌重复计数;裁判语义为"其余 3 张已亮明"。
2. 抢杠"和绝张":裁判从不计(BUGANG 时 shownTile 已置 4,judge 条件 ==3 永假)。
3. 组合龙 + 边张/嵌张/单钓将:和牌张在组合龙范围内时裁判不计该 1 番
   (fan_calculator.cpp L1842),PyMahjongGB 1.3.0 会多算。
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_BASELINE_RL = os.path.normpath(os.path.join(_SRC, '..', 'baseline', 'RL'))
if _BASELINE_RL not in sys.path:
    sys.path.insert(0, _BASELINE_RL)
from env import MahjongGBEnv, Error  # noqa: E402
from feature_v1 import judge_fan_total  # noqa: E402
from MahjongGB import MahjongFanCalculator  # noqa: E402


class MahjongGBEnvFixed(MahjongGBEnv):

    def _checkMahjong(self, player, isSelfDrawn=False, isAboutKong=False):
        # 逻辑复刻 baseline _checkMahjong,差异仅:is4thTile 语义对齐裁判、
        # 总番经 judge_fan_total 修正
        shown = self.shownTiles[self.curTile]
        if isSelfDrawn or isAboutKong:
            # 自摸:step() 已提前 +1,抵消重复计数;抢杠:裁判永不计和绝张
            shown -= 1
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[player]),
                hand=tuple(self.hands[player]),
                winTile=self.curTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(shown + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=player,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = judge_fan_total(fans, self.curTile, self.hands[player])
            if fanCnt < 8:
                raise Error('Not Enough Fans')
            self.obs = {i: self.agents[i].request2obs('Player %d Hu' % player) for i in range(4)}
            if isSelfDrawn:
                self.reward = [-(8 + fanCnt)] * 4
                self.reward[player] = (8 + fanCnt) * 3
            else:
                self.reward = [-8] * 4
                self.reward[player] = 8 * 3 + fanCnt
                self.reward[self.curPlayer] -= fanCnt
            self.done = True
        except Exception:
            raise Error(player)
