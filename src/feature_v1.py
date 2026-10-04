"""特征 v1:FeatureAgentV1。

观测 102×4×9 uint8(通道布局见 FEATURE.md),动作空间 235(与 baseline 完全一致)。
状态机骨架沿用 baseline/RL/feature.py(含 UnChi/UnPeng,供 Botzone 两段式决策使用),
观测在 _obs() 时整体渲染(非增量);向听通道用 PyMahjongGB 的 shanten 接口。

本类是唯一实现,被预处理(replay_driver / preprocess_v1)、RL 环境(baseline env)、
未来 Botzone 部署三处共用。
"""
import os
import sys
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

_BASELINE_RL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baseline', 'RL'))
if os.path.isdir(_BASELINE_RL) and _BASELINE_RL not in sys.path:
    sys.path.insert(0, _BASELINE_RL)
try:
    from agent import MahjongGBAgent  # noqa: E402
except ImportError:
    class MahjongGBAgent(object):
        """Botzone 部署包内无 baseline,提供最小基类(env 不在部署路径上)。"""
        observation_space = None
        action_space = None

try:
    from MahjongGB import MahjongFanCalculator
except ImportError:
    print('MahjongGB library required! pip install PyMahjongGB')
    raise

import shanten_py  # 自研向听引擎(库的 shanten 接口有状态污染 bug,见 shanten_py.py 头注)

FEATURE_VERSION = "v1"

# ---------------- 通道布局(与 FEATURE.md 严格同步) ----------------
CH_HAND = 0             # 4  自己手牌计数温度计 >=1/>=2/>=3/>=4
CH_DRAWN = 4            # 1  刚摸到的牌(仅出牌决策时)
CH_MELD = 5             # 20 副露:每相对家 [计数温度计4 + 暗杠数广播1] x4(自/下/对/上)
CH_DISCARD_ALL = 25     # 16 弃牌全集温度计 4 x4 家
CH_DISCARD_RECENT = 41  # 32 最近8张弃牌 one-hot(新->旧)x4 家
CH_CUR_TILE = 73        # 1  待响应的牌
CH_CUR_OFFER = 74       # 3  出牌者相对位置广播(下家/对家/上家)
CH_UNSEEN = 77          # 4  未见牌计数温度计
CH_SEAT_WIND = 81       # 4  门风 one-hot 广播
CH_PREV_WIND = 85       # 4  圈风 one-hot 广播
CH_WALL_TOTAL = 89      # 1  牌墙总剩余/84, x255 存储
CH_WALL_SELF = 90       # 1  自己剩余摸牌/21, x255 存储
CH_SHANTEN = 91         # 7  向听数 one-hot 0..6+
CH_USEFUL = 98          # 1  有效进张牌集 mask
CH_USEFUL_UNSEEN = 99   # 1  有效进张未见总张数/136, x255 存储
CH_WALL_LAST = 100      # 1  海底(本轮打出后无人能再摸牌)
CH_ABOUT_KONG = 101     # 1  当前决策与杠相关(岭上摸牌/抢杠在决策点)
NUM_CHANNELS = 102

# x255 存储、loader 侧需 /255 还原的标量通道
SCALED_CHANNELS = (CH_WALL_TOTAL, CH_WALL_SELF, CH_USEFUL_UNSEEN)


# ---------------- 向听数与有效进张(模块级,带缓存,RuleBot 复用) ----------------
_CACHE_WAIT = int(os.environ.get('SHANTEN_CACHE_WAIT', 65536))
_CACHE_DISCARD = int(os.environ.get('SHANTEN_CACHE_DISCARD', 65536))


# 组合龙 6 种花色指派对应的 9 张牌集合(用于裁判语义的番数修正)
_KNIT_TILE_SETS = [frozenset(shanten_py.TILE_LIST[i] for i in pat)
                   for pat in shanten_py._KNIT_PATTERNS]


def judge_fan_total(fans, winTile, hand_tiles):
    """按官方裁判语义计算总番。

    裁判内嵌的 fan-calculator(judge 仓库 fan_calculator.cpp L1842)规定:
    "和牌张是组合龙范围的牌,不计边张、嵌张、单钓将";PyMahjongGB 1.3.0 缺
    这条规则会多算 1 番(全量对拍 17/95627 例不一致全部源于此)。
    和牌张在组合龙范围外时,边张/嵌张/单钓将正常计(数据中有 106 例佐证)。

    fans: MahjongFanCalculator(verbose=True) 的 (point, cnt, 中文名, 英文名) 列表;
    hand_tiles: 和牌时刻手牌(不含 winTile)。
    """
    total = sum(f * c for f, c, _, _ in fans)
    names = {n for _, _, n, _ in fans}
    if '组合龙' in names and names & {'边张', '嵌张', '单钓将'}:
        tiles = set(hand_tiles) | {winTile}
        for pat in _KNIT_TILE_SETS:
            if winTile in pat and pat <= tiles:
                total -= 1
                break
    return total


@lru_cache(maxsize=_CACHE_WAIT)
def shanten_useful_wait(hand):
    """hand: 已排序 tuple,长度 % 3 == 1(13/10/7/4/1,副露已扣除)。

    返回 (向听数 0..6, 有效进张 frozenset)。综合常规与特殊牌型
    (七对/十三幺/全不靠仅 13 张可算,组合龙 13/10 张),取最小向听,
    有效进张为所有达到最小向听的牌型的进张并集。
    """
    results = []
    try:
        results.append(shanten_py.regular_shanten_useful(hand))
    except Exception:
        pass
    if len(hand) == 13:
        for fn in (shanten_py.seven_pairs_shanten_useful,
                   shanten_py.thirteen_orphans_shanten_useful,
                   shanten_py.honors_knitted_shanten_useful):
            try:
                results.append(fn(hand))
            except Exception:
                pass
    if len(hand) in (13, 10):
        try:
            results.append(shanten_py.knitted_straight_shanten_useful(hand))
        except Exception:
            pass
    if not results:
        return 6, frozenset()
    best = min(r[0] for r in results)
    useful = set()
    for s, u in results:
        if s == best:
            useful.update(u)
    return min(best, 6), frozenset(useful)


@lru_cache(maxsize=_CACHE_DISCARD)
def shanten_useful_discard(hand):
    """hand: 已排序 tuple,长度 % 3 == 2(14/11/8/5/2,待打出一张)。

    定义为对所有不同弃牌取最小的 13 张向听;有效进张为达到最小向听的弃法的进张并集。
    """
    best, useful = 99, set()
    hand_list = list(hand)
    for t in set(hand_list):
        rest = list(hand_list)
        rest.remove(t)
        s, u = shanten_useful_wait(tuple(sorted(rest)))
        if s < best:
            best, useful = s, set(u)
        elif s == best:
            useful |= u
    return min(best, 6), frozenset(useful)


class FeatureAgentV1(MahjongGBAgent):

    OBS_SIZE = NUM_CHANNELS
    ACT_SIZE = 235

    observation_space = (NUM_CHANNELS, 4, 9)
    action_space = 235

    OFFSET_ACT = {
        'Pass': 0,
        'Hu': 1,
        'Play': 2,
        'Chi': 36,
        'Peng': 99,
        'Gang': 133,
        'AnGang': 167,
        'BuGang': 201
    }
    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),
        *('T%d' % (i + 1) for i in range(9)),
        *('B%d' % (i + 1) for i in range(9)),
        *('F%d' % (i + 1) for i in range(4)),
        *('J%d' % (i + 1) for i in range(3))
    ]
    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.prevalentWind = -1
        self.hand = []
        self.packs = [[] for _ in range(4)]      # 相对座位:0自 1下 2对 3上
        self.history = [[] for _ in range(4)]    # 每家弃牌序列(含被吃碰杠走的)
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.lastDrawn = None
        self.curTile = None
        self.tileFrom = -1
        self.valid = []
        # 渲染上下文:'play'(需打牌)/'claim'(响应他家打牌)/'rob'(响应补杠)/'terminal'
        self._mode = 'terminal'
        self._aboutKongRender = False

    # ---------------- 事件状态机(骨架与 baseline/RL/feature.py 一致) ----------------
    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            return
        if t[0] == 'Huang':
            self.valid = []
            self._mode = 'terminal'
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            aboutKong = self.isAboutKong
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=aboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self.lastDrawn = tile
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            self._mode = 'play'
            self._aboutKongRender = aboutKong
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            self._mode = 'terminal'
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            self._mode = 'terminal'
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self.lastDrawn = None
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3):
                            tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                self._mode = 'claim'
                self._aboutKongRender = False
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self.lastDrawn = None
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                self._mode = 'play'
                self._aboutKongRender = False
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self.lastDrawn = None
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                self._mode = 'play'
                self._aboutKongRender = False
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self.lastDrawn = None
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                self.lastDrawn = None
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self.lastDrawn = None
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                # 抢杠和:官方裁判不计"和绝张"(judge/main.cpp 在 BUGANG 时已置
                # shownTile=4,查胡条件 ==3 永假),故强制 is4thTile=False
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True,
                                       is4thTile=False):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                self.curTile = tile
                self.tileFrom = p
                self._mode = 'rob'
                self._aboutKongRender = True
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)

    # ---------------- 动作编解码(与 baseline 完全一致) ----------------
    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass':
            return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu':
            return self.OFFSET_ACT['Hu']
        if t[0] == 'Play':
            return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi':
            return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng':
            return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang':
            return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang':
            return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang':
            return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    # ---------------- 观测渲染 ----------------
    def _unseen_counts(self):
        """每种牌从自己视角的未见张数(0..4)。其他家暗杠的牌未知,无法扣除。"""
        unseen = np.full(34, 4, dtype=np.int32)
        for tile in self.hand:
            unseen[self.OFFSET_TILE[tile]] -= 1
        for tile, c in self.shownTiles.items():
            if c:
                unseen[self.OFFSET_TILE[tile]] -= c
        for packType, tile, offer in self.packs[0]:
            if packType == 'GANG' and offer == 0 and tile != 'CONCEALED':
                unseen[self.OFFSET_TILE[tile]] -= 4  # 自己的暗杠不在 shownTiles 中
        np.clip(unseen, 0, 4, out=unseen)
        return unseen

    def _obs(self):
        OT = self.OFFSET_TILE
        obs = np.zeros((self.OBS_SIZE, 36), dtype=np.uint8)
        # 1. 手牌温度计
        for tile, c in Counter(self.hand).items():
            obs[CH_HAND:CH_HAND + min(c, 4), OT[tile]] = 1
        # 2. 刚摸到的牌
        if self._mode == 'play' and self.lastDrawn is not None:
            obs[CH_DRAWN, OT[self.lastDrawn]] = 1
        # 3. 四家副露
        for p in range(4):
            base = CH_MELD + 5 * p
            angang = 0
            mc = Counter()
            for packType, tile, offer in self.packs[p]:
                if packType == 'PENG':
                    mc[tile] += 3
                elif packType == 'GANG':
                    if tile == 'CONCEALED':
                        angang += 1
                    else:
                        mc[tile] += 4
                        if offer == 0:
                            angang += 1  # 自己的暗杠(牌面可见于自己)
                elif packType == 'CHI':
                    color, num = tile[0], int(tile[1])
                    for d in (-1, 0, 1):
                        mc[color + str(num + d)] += 1
            for tile, c in mc.items():
                obs[base:base + min(c, 4), OT[tile]] = 1
            if angang:
                obs[base + 4, :] = angang
        # 4./5. 弃牌全集 + 最近8张
        for p in range(4):
            for tile, c in Counter(self.history[p]).items():
                obs[CH_DISCARD_ALL + 4 * p:CH_DISCARD_ALL + 4 * p + min(c, 4), OT[tile]] = 1
            for k, tile in enumerate(self.history[p][::-1][:8]):
                obs[CH_DISCARD_RECENT + 8 * p + k, OT[tile]] = 1
        # 6. 待响应的牌
        if self._mode in ('claim', 'rob'):
            obs[CH_CUR_TILE, OT[self.curTile]] = 1
            obs[CH_CUR_OFFER + self.tileFrom - 1, :] = 1
        # 7. 未见牌温度计
        unseen = self._unseen_counts()
        for i in range(34):
            c = unseen[i]
            if c > 0:
                obs[CH_UNSEEN:CH_UNSEEN + c, i] = 1
        # 8. 门风/圈风
        obs[CH_SEAT_WIND + self.seatWind, :] = 1
        if self.prevalentWind >= 0:
            obs[CH_PREV_WIND + self.prevalentWind, :] = 1
        # 9. 进度
        obs[CH_WALL_TOTAL, :] = int(round(max(sum(self.tileWall), 0) / 84.0 * 255))
        obs[CH_WALL_SELF, :] = int(round(max(self.tileWall[0], 0) / 21.0 * 255))
        # 10. 向听特征(终局状态跳过)
        if self._mode != 'terminal':
            n = len(self.hand)
            try:
                if n % 3 == 2:
                    sh, useful = shanten_useful_discard(tuple(sorted(self.hand)))
                elif n % 3 == 1:
                    sh, useful = shanten_useful_wait(tuple(sorted(self.hand)))
                else:
                    sh, useful = None, ()
            except Exception:
                sh, useful = 6, ()
            if sh is not None:
                obs[CH_SHANTEN + min(sh, 6), :] = 1
                uu = 0
                for tile in useful:
                    obs[CH_USEFUL, OT[tile]] = 1
                    uu += unseen[OT[tile]]
                obs[CH_USEFUL_UNSEEN, :] = int(round(min(uu, 136) / 136.0 * 255))
        # 11. 杂项
        if self.wallLast:
            obs[CH_WALL_LAST, :] = 1
        if self._aboutKongRender:
            obs[CH_ABOUT_KONG, :] = 1

        mask = np.zeros(self.ACT_SIZE, dtype=np.uint8)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': obs.reshape((self.OBS_SIZE, 4, 9)),
            'action_mask': mask
        }

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False, is4thTile=None):
        if is4thTile is None:
            is4thTile = (self.shownTiles[winTile] + isSelfDrawn) == 4
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=is4thTile,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = judge_fan_total(fans, winTile, self.hand)
            if fanCnt < 8:
                raise Exception('Not Enough Fans')
        except Exception:
            return False
        return True
