"""自对弈牌谱导出:把 env_fixed 的一局还原成 data.txt 格式行(供门禁2与离线分析)。

不输出 Ignore 子句(只记录被执行的宣告;obs 重建不受影响)。
用法:logger = GameLogger(env)(reset 之后构造);每次 env.step 前调
pre_step(action_dict),step 后调 post_step(rewards, done)。
"""


class GameLogger:

    def __init__(self, env, wind):
        self.env = env
        self.lines = ['Wind %d' % wind]
        for i in range(4):
            self.lines.append('Player %d Deal %s' % (i, ' '.join(env.hands[i])))
        self._log_draw()

    def _log_draw(self):
        self.lines.append('Player %d Draw %s' % (self.env.curPlayer, self.env.curTile))

    def pre_step(self, action_dict):
        env = self.env
        self._pre_state = env.state
        self._pre_player = env.curPlayer
        self._pre_tile = env.curTile
        self._responses = {}
        for name, a in action_dict.items():
            seat = int(name.split('_')[1]) - 1
            self._responses[seat] = env.agents[seat].action2response(a)

    def post_step(self, rewards, done):
        env = self.env
        st, p0, tile0 = self._pre_state, self._pre_player, self._pre_tile
        r = self._responses
        if st in (0, 1):
            resp = r[p0].split()
            if resp[0] == 'Play':
                self.lines.append('Player %d Play %s' % (p0, resp[1]))
            elif resp[0] == 'Hu':
                self.lines.append('Player %d Hu %s' % (p0, tile0))
            elif resp[0] == 'Gang':   # 暗杠(state1 的 Gang XX)
                self.lines.append('Player %d AnGang %s' % (p0, resp[1]))
                if not done:
                    self._log_draw()
            elif resp[0] == 'BuGang':
                self.lines.append('Player %d BuGang %s' % (p0, resp[1]))
                # 进入抢杠窗口,摸牌在下一步(若无人抢)
        elif st == 2:
            # 响应弃牌:看 env 实际执行了什么
            executed = None
            for j in range(1, 4):
                q = (p0 + j) % 4
                resp = r.get(q, 'Pass').split()
                if resp[0] == 'Hu':
                    executed = ('Hu', q, tile0)
                    break
            if executed is None:
                for j in range(1, 4):
                    q = (p0 + j) % 4
                    resp = r.get(q, 'Pass').split()
                    if resp[0] == 'Gang':
                        executed = ('Gang', q, tile0)
                        break
                    if resp[0] == 'Peng':
                        executed = ('Peng', q, tile0)
                        break
            if executed is None:
                q = (p0 + 1) % 4
                resp = r.get(q, 'Pass').split()
                if resp[0] == 'Chi':
                    executed = ('Chi', q, resp[1])
            if executed:
                self.lines.append('Player %d %s %s' % (executed[1], executed[0], executed[2]))
                if executed[0] == 'Gang' and not done:
                    self._log_draw()
            elif not done:
                self._log_draw()      # 全过,下家摸牌
        elif st == 3:
            hu_q = None
            for j in range(1, 4):
                q = (p0 + j) % 4
                if r.get(q, 'Pass').split()[0] == 'Hu':
                    hu_q = q
                    break
            if hu_q is not None:
                self.lines.append('Player %d Hu %s' % (hu_q, tile0))
            elif not done:
                self._log_draw()      # 无人抢杠,补杠者摸岭上

        if done:
            if max(rewards) == 0:
                self.lines.append('Huang')
            self.lines.append('Score %d %d %d %d' % tuple(rewards))
        return self.lines
