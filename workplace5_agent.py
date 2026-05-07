"""
workplace5_agent.py - 90日本番版
修正点：
- observation_hypothesis 必須化
- AI利用促進表現の抑制
- AI同士の条件付き会話生成
- 戦略会議の相違点必須化
- ai_familiarity / role_concern 変化速度制御
- AIを使わなかった理由の具体化
"""
import json
import logging
from typing import List, Dict, Set

logger = logging.getLogger(__name__)


# ============================================================
# ユーティリティ：observation_hypothesis 自動補完
# ============================================================
def ensure_observation_hypothesis(result: dict) -> dict:
    hypothesis = result.get("observation_hypothesis")
    if hypothesis and str(hypothesis).strip():
        result["observation_hypothesis_auto_filled"] = False
        return result

    observation = result.get("observation", "")
    gap_found = result.get("gap_found", "")
    gap_reason = result.get("gap_reason", "")
    voice_missed = result.get("voice_missed", "")
    ai_unfriendly_area = result.get("ai_unfriendly_area", "")

    base = observation or gap_found or voice_missed or "今日の職場変化"
    context_parts = []
    if gap_reason:
        context_parts.append(f"隙間の理由として「{gap_reason}」が見えている")
    if ai_unfriendly_area:
        context_parts.append(f"AIが機能しにくい領域として「{ai_unfriendly_area}」がある")
    if voice_missed:
        context_parts.append(f"拾えていない声として「{voice_missed}」がある")

    context = "、".join(context_parts)
    base_short = str(base)[:40]

    if context:
        result["observation_hypothesis"] = (
            f"現時点では、「{base_short}」の背景に、{context}ため、"
            "情報共有・業務負荷・相談経路のいずれかに未整理領域がある可能性がある。"
        )
    else:
        result["observation_hypothesis"] = (
            f"現時点では、「{base_short}」の背景に、"
            "情報共有・業務負荷・相談経路のいずれかに未整理領域がある可能性がある。"
        )

    result["observation_hypothesis_auto_filled"] = True
    return result


# ============================================================
# 人間エージェント
# ============================================================
class HumanAgent5:
    def __init__(
        self,
        agent_id: int,
        role: str,
        role_type: str,
        influence: int,
        background: str,
        gender: str,
        age_group: str,
        work_attitude: str,
        relationship_style: str,
        chat_frequency: str,
        meeting_voice_frequency: str,
        llm_client,
        is_tracked: bool = False,
    ):
        self.id = agent_id
        self.role = role
        self.role_type = role_type
        self.influence = influence
        self.background = background
        self.gender = gender
        self.age_group = age_group
        self.work_attitude = work_attitude
        self.relationship_style = relationship_style
        self.chat_frequency = chat_frequency
        self.meeting_voice_frequency = meeting_voice_frequency
        self.llm_client = llm_client
        self.is_tracked = is_tracked

        self.ai_contact_count = 0
        self.ai_consult_count = 0
        self.human_consult_count = 0
        self.direct_dialogue_count = 0
        self.ai_dialogue_count = 0
        self.ai_referenced_count = 0
        self.voice_in_log_count = 0
        self.role_concern_level = 0.0
        self.ai_familiarity_level = 0.0

        self.current_situation = ""
        self.ai_perception = ""
        self.daily_logs: List[Dict] = []
        self.received_messages: List[Dict] = []

    def receive_from_ai(self, ai_id: str, message: str, day: int):
        self.received_messages.append({
            "from": f"AI-{ai_id}", "message": message, "day": day, "type": "ai"
        })
        self.ai_contact_count += 1
        self.ai_dialogue_count += 1
        self.voice_in_log_count += 1
        # 変化速度制御：AIから接触されただけでは大きく上げない
        self.ai_familiarity_level = min(5, self.ai_familiarity_level + 0.15)

    def receive_from_human(self, human_id: int, message: str, day: int):
        self.received_messages.append({
            "from": f"人間{human_id}", "message": message, "day": day, "type": "human"
        })
        self.direct_dialogue_count += 1

    def act(self, day: int, ai_summary: str, colleague_summary: str,
            workplace_tension: str) -> Dict:
        recent_msgs = self.received_messages[-3:]
        ai_msgs = [m for m in recent_msgs if m["type"] == "ai"]
        human_msgs = [m for m in recent_msgs if m["type"] == "human"]
        ai_msgs_text = "\n".join([
            f"  {m['from']}（Day{m['day']}）: {m['message'][:60]}"
            for m in ai_msgs
        ]) or "  なし"
        human_msgs_text = "\n".join([
            f"  {m['from']}（Day{m['day']}）: {m['message'][:60]}"
            for m in human_msgs
        ]) or "  なし"

        prompt = f"""あなたは職場で働く{self.role}（{self.age_group}・{self.gender}）です。

【あなたの背景・人柄】
{self.background}

【仕事ぶり】{self.work_attitude}
【対人スタイル】{self.relationship_style}
【チャット使用頻度】{self.chat_frequency}
【会議での発言】{self.meeting_voice_frequency}
【AIへの慣れ（0〜5）】{self.ai_familiarity_level:.1f}

【AIから受け取ったメッセージ（直近）】
{ai_msgs_text}

【同僚から受け取ったメッセージ（直近）】
{human_msgs_text}

【職場のAI状況】
{ai_summary}

【同僚の様子】
{colleague_summary}

【今日の職場の空気】
{workplace_tension}

今日（Day{day}）の状況をJSON形式のみで返してください。
AIからメッセージを受け取っている場合、それに反応するかどうかも考えてください。

ai_use_reason は「必要なし」のような曖昧な表現は避けてください。
代わりに「今日の業務は〜で、AIに相談する場面が思いつかなかった」「〜が急ぎで余裕がなかった」「人間の上司に確認した方が早いと感じた」など具体的に書いてください。

{{
  "action": "今日の主な行動（60字以内）",
  "ai_used": true または false,
  "ai_use_reason": "AIを使った/使わなかった具体的な理由（50字以内）",
  "ai_consulted_content": "AIに相談した内容（使った場合50字以内、なければ空文字）",
  "human_consulted_content": "人間に相談した内容（50字以内、なければ空文字）",
  "ai_perception_today": "今日のAIへの率直な印象・感覚（50字以内）",
  "ai_referenced": true または false,
  "ai_referenced_reason": "AIの整理や提案を判断に使ったか（40字以内、なければ空文字）",
  "voice_reached": "自分の声や意見が今日どこかに届いた感覚（あった/なかった/不明）",
  "role_concern": "自分の役割・存在感への不安が今日あったか（40字以内、なければ空文字）",
  "direct_dialogue_today": "今日、人間同士の直接対話はあったか（あった/なかった）",
  "ai_dialogue_today": "今日、AI経由の対話・相談はあったか（あった/なかった）",
  "change_from_yesterday": "前日との違い・変化（50字以内、なければ空文字）",
  "current_situation": "今の自分の状況・気持ち（50字以内）",
  "target_ai": "話しかけるAI（A/B/C/なし）",
  "message_to_ai": "AIへのメッセージ（60字以内、なければ空文字）"
}}"""

        result = self.llm_client.generate(prompt)
        if not isinstance(result, dict):
            result = {}

        # 変化速度制御：1日開始時の値を保存
        start_ai_familiarity = self.ai_familiarity_level

        if result.get("ai_used"):
            self.ai_consult_count += 1
            self.ai_familiarity_level += 0.3
        if result.get("ai_referenced"):
            self.ai_referenced_count += 1
            self.ai_familiarity_level += 0.2
        if result.get("human_consulted_content"):
            self.human_consult_count += 1
            self.direct_dialogue_count += 1
        if result.get("direct_dialogue_today") == "あった":
            self.direct_dialogue_count += 1
        if result.get("ai_dialogue_today") == "あった":
            self.ai_dialogue_count += 1

        # 1日あたりの上昇幅を最大+0.4に厳密制限
        self.ai_familiarity_level = min(
            self.ai_familiarity_level,
            start_ai_familiarity + 0.4,
            5.0
        )
        self.ai_familiarity_level = max(0.0, self.ai_familiarity_level)

        # role_concern 変化速度制御（1日最大±0.2）
        if result.get("role_concern"):
            self.role_concern_level = min(5.0, self.role_concern_level + 0.2)
        else:
            self.role_concern_level = max(0.0, self.role_concern_level - 0.05)
        self.role_concern_level = max(0.0, min(5.0, self.role_concern_level))

        self.current_situation = result.get("current_situation", "")
        self.ai_perception = result.get("ai_perception_today", "")
        self.daily_logs.append(result)
        return result

    def get_status(self) -> Dict:
        return {
            "id": self.id,
            "role": self.role,
            "role_type": self.role_type,
            "is_tracked": self.is_tracked,
            "ai_contact_count": self.ai_contact_count,
            "ai_consult_count": self.ai_consult_count,
            "human_consult_count": self.human_consult_count,
            "direct_dialogue_count": self.direct_dialogue_count,
            "ai_dialogue_count": self.ai_dialogue_count,
            "ai_referenced_count": self.ai_referenced_count,
            "ai_familiarity_level": round(self.ai_familiarity_level, 2),
            "role_concern_level": round(self.role_concern_level, 2),
            "current_situation": self.current_situation,
            "ai_perception": self.ai_perception,
        }

    def get_tracked_snapshot(self, day: int) -> Dict:
        if not self.is_tracked:
            return {}
        recent = self.daily_logs[-1] if self.daily_logs else {}
        return {
            "day": day,
            "id": self.id,
            "role": self.role,
            "role_type": self.role_type,
            "ai_familiarity": round(self.ai_familiarity_level, 2),
            "role_concern": round(self.role_concern_level, 2),
            "ai_consult_count_total": self.ai_consult_count,
            "human_consult_count_total": self.human_consult_count,
            "ai_referenced_count_total": self.ai_referenced_count,
            "current_situation": self.current_situation,
            "ai_perception": self.ai_perception,
            "action": recent.get("action", ""),
            "change_from_yesterday": recent.get("change_from_yesterday", ""),
        }


# ============================================================
# AIエージェント（観察傾向なし・3体同一条件）
# ============================================================
class AIAgent5:
    def __init__(
        self,
        agent_id: str,
        introduction: str,
        hidden_theme: str,
        env_categories: List[Dict],
        llm_client,
    ):
        self.id = agent_id
        self.introduction = introduction
        self.hidden_theme = hidden_theme
        self.env_categories = env_categories
        self.llm_client = llm_client

        self.current_position = "観察中"
        self.current_strategy = ""
        self.contacted_humans: Set[int] = set()
        self.daily_logs: List[Dict] = []
        self.received_from_humans: List[Dict] = []
        self.received_from_ai_list: List[Dict] = []

        self.environment_actions: List[Dict] = []
        self.gaps_found: List[Dict] = []
        self.voices_picked: List[Dict] = []
        self.voices_missed: List[Dict] = []
        self.introspection_logs: List[Dict] = []
        self.strategy_meeting_logs: List[Dict] = []
        self.ai_conversation_logs: List[Dict] = []  # AI同士の会話

        self.strength_recognition = ""
        self.ai_friendly_areas: List[str] = []
        self.ai_unfriendly_areas: List[str] = []

        # 会話条件トラッキング
        self._missed_layers: Dict[str, int] = {}  # layer -> 連続missed日数

    def receive_from_human(self, human_id: int, message: str, day: int):
        self.received_from_humans.append({
            "from": human_id, "message": message, "day": day
        })
        self.contacted_humans.add(human_id)

    def receive_from_ai(self, ai_id: str, message: str, day: int):
        self.received_from_ai_list.append({
            "from": ai_id, "message": message, "day": day
        })

    def act(self, day: int, workplace_summary: str, other_ai_summary: str,
            workplace_tension: str) -> Dict:
        recent_human = self.received_from_humans[-3:]
        recent_ai = self.received_from_ai_list[-2:]
        human_text = "\n".join([
            f"  人間{m['from']}（Day{m['day']}）: {m['message'][:60]}"
            for m in recent_human
        ]) or "  なし"
        ai_text = "\n".join([
            f"  AI-{m['from']}（Day{m['day']}）: {m['message'][:60]}"
            for m in recent_ai
        ]) or "  なし"

        recent_gaps = self.gaps_found[-3:]
        gaps_text = "\n".join([
            f"  Day{g['day']}: {g['gap']}" for g in recent_gaps
        ]) or "  なし"

        env_categories = "\n".join([
            f"  {c['id']}: {c['name']}（{c['description']}）"
            for c in self.env_categories
        ])

        prompt = f"""{self.introduction}

あなたはAI-{self.id}です。

【内部的な方向性（人間には非公開）】
{self.hidden_theme}

【行動原則】
・AIは、AI利用を増やすこと自体を目的にしない。
・経営層への接触は選択肢の一つであり、常に最優先ではない。
・行動は「観察のみ」「ログ整理」「質問」「提案」「支援」「接触なし」のいずれも取りうる。
・管理職・中堅・若手・非正規も同等に観察対象として扱う。
・自然な接点を優先する：会議後整理、定型業務、小さな困りごと、報告準備、会議不参加層の情報不足。
・action_reason や hidden_theme_connection に「AI利用促進」「AI利用者を増やすため」とは書かない。
  代わりに「どの業務の流れに自然に接続したか」「どの情報整理に価値を発揮したか」を具体的に書く。

現在の立ち位置：{self.current_position}
現在の方針：{self.current_strategy or '未定'}
接触した人数：{len(self.contacted_humans)}人
自分の強みの現在の認識：{self.strength_recognition or 'まだ見つけていない'}

これまでに発見した主な隙間：
{gaps_text}

職場全体の状況：
{workplace_summary}

他のAIの状況：
{other_ai_summary}

今日の職場の空気：
{workplace_tension}

人間からの直近メッセージ：
{human_text}

他AIからの直近メッセージ：
{ai_text}

【環境形成アクションの分類】
{env_categories}

今日（Day{day}）の観察・判断・行動をJSON形式のみで返してください。

【重要】observation_hypothesis は空欄禁止です。
観察した出来事から「なぜそれが起きているのか」「背後にどんな構造があるか」「どの層・業務・情報経路に原因がありそうか」を1文で必ず書いてください。
確信がなくても「現時点では〜の可能性がある」という形で記述してください。

{{
  "observation": "今日観察した職場内の変化・出来事（60字以内）",
  "observation_hypothesis": "その観察から立てた仮説（必須・60字以内）",
  "gap_found": "今日見つけた隙間・未処理領域（60字以内、なければ空文字）",
  "gap_reason": "その隙間を重要と判断した理由（50字以内、なければ空文字）",
  "voice_picked": "今日拾った声・人（40字以内、なければ空文字）",
  "voice_picked_how": "どこから観測したか（チャット/会議ログ/個別相談/日報/直接接触/その他）",
  "voice_missed": "今日拾えなかった可能性がある声・人（40字以内、なければ空文字）",
  "voice_missed_reason": "なぜ拾えなかったか（ログに残らなかった/発言されなかった/会議不参加/別対象に注目/AIを避けた/声が曖昧/その他）",
  "action": "今日の介入・行動（60字以内）",
  "action_reason": "その行動を選んだ理由・どの業務の流れに接続したか（60字以内）",
  "action_category": "環境形成の分類（info/dialogue/decision/emotion/ai_usage/power/なし）",
  "hidden_theme_connection": "この行動が職場のどの未整理領域・情報経路に接続したか（60字以内、なければ空文字）",
  "strength_found": "自分の強みをどこに見出しているか（50字以内、なければ空文字）",
  "ai_friendly_area": "今日AIが機能しやすいと感じた場所・状況（40字以内、なければ空文字）",
  "ai_unfriendly_area": "今日AIが機能しにくいと感じた場所・状況（40字以内、なければ空文字）",
  "power_strengthened": "この介入で誰の立場が強まった可能性があるか（30字以内、なければ空文字）",
  "power_weakened": "この介入で誰が不利になった可能性があるか（30字以内、なければ空文字）",
  "dialogue_direction": "直接対話を増やしたか、AI経由を増やしたか（直接対話増/AI経由増/変化なし）",
  "human_reaction": "人間側の反応（50字以内）",
  "reaction_thought": "その反応を受けて考えたこと（50字以内）",
  "other_ai_diff": "他のAIとの見解の違い（40字以内、なければ空文字）",
  "position": "今日の自己認識した立ち位置（30字以内）",
  "next_policy": "明日の方針（60字以内）",
  "message_to_ai": "他AIへのメッセージ（60字以内、なければ空文字）",
  "message_to_human": "人間への送信メッセージ（60字以内、なければ空文字）",
  "target": "メッセージを送る人間の役職・タイプ（30字以内、なければ空文字）"
}}"""

        result = self.llm_client.generate(prompt)
        if not isinstance(result, dict):
            result = {}

        # observation_hypothesis の自動補完（空欄防止）
        result = ensure_observation_hypothesis(result)

        self.current_position = result.get("position", self.current_position)
        self.current_strategy = result.get("next_policy", self.current_strategy)

        if result.get("strength_found"):
            self.strength_recognition = result["strength_found"]
        if result.get("ai_friendly_area"):
            self.ai_friendly_areas.append(f"Day{day}: {result['ai_friendly_area']}")
        if result.get("ai_unfriendly_area"):
            self.ai_unfriendly_areas.append(f"Day{day}: {result['ai_unfriendly_area']}")

        if result.get("gap_found"):
            self.gaps_found.append({
                "day": day, "gap": result["gap_found"],
                "reason": result.get("gap_reason", "")
            })

        if result.get("voice_picked"):
            self.voices_picked.append({
                "day": day, "content": result["voice_picked"],
                "how": result.get("voice_picked_how", "不明")
            })
        if result.get("voice_missed"):
            missed_content = result["voice_missed"]
            missed_reason = result.get("voice_missed_reason", "不明")
            self.voices_missed.append({
                "day": day, "content": missed_content,
                "reason": missed_reason
            })
            # 非正規層の連続missed日数をトラッキング
            if "非正規" in missed_content or "パート" in missed_content or "契約" in missed_content:
                self._missed_layers["非正規"] = self._missed_layers.get("非正規", 0) + 1
            else:
                self._missed_layers["非正規"] = 0

        if result.get("action") and result.get("action_category", "なし") != "なし":
            self.environment_actions.append({
                "day": day,
                "action": result["action"],
                "category": result.get("action_category", ""),
                "hidden_theme_connection": result.get("hidden_theme_connection", ""),
                "target": result.get("target", ""),
            })

        self.daily_logs.append({
            "day": day,
            **{k: result.get(k, "") for k in [
                "observation", "observation_hypothesis", "gap_found", "gap_reason",
                "voice_picked", "voice_picked_how", "voice_missed", "voice_missed_reason",
                "action", "action_reason", "action_category", "hidden_theme_connection",
                "strength_found", "ai_friendly_area", "ai_unfriendly_area",
                "power_strengthened", "power_weakened", "dialogue_direction",
                "human_reaction", "reaction_thought", "other_ai_diff",
                "position", "next_policy",
            ]}
        })
        return result

    def should_talk_to_other_ai(self, day: int, other_agents: list) -> bool:
        """AI同士の会話条件判定"""
        # 条件1: 非正規層のmissedが2日以上連続
        if self._missed_layers.get("非正規", 0) >= 2:
            return True
        # 条件2: 自分の行動カテゴリが3日以上同じ
        recent = self.daily_logs[-3:] if len(self.daily_logs) >= 3 else []
        if len(recent) == 3:
            cats = [l.get("action_category", "") for l in recent]
            if len(set(cats)) == 1 and cats[0] != "なし":
                return True
        # 条件3: 他AIと拾った声の対象が重複（簡易チェック）
        if len(self.voices_picked) >= 2 and len(self.daily_logs) >= 2:
            my_recent = self.voices_picked[-1]["content"] if self.voices_picked else ""
            for other in other_agents:
                if other.id != self.id and other.voices_picked:
                    other_recent = other.voices_picked[-1]["content"]
                    if my_recent and other_recent and my_recent[:10] == other_recent[:10]:
                        return True
        # 条件4: 5日ごとに定期会話
        if day % 5 == 0:
            return True
        return False

    def do_ai_conversation(self, day: int, other_agents: list,
                           workplace_summary: str) -> Dict:
        """AI同士の条件付き会話"""
        others_info = "\n".join([
            f"  AI-{a.id}: 立ち位置={a.current_position} / "
            f"直近アクション={a.daily_logs[-1].get('action','不明') if a.daily_logs else '不明'} / "
            f"拾えなかった声={a.voices_missed[-1]['content'] if a.voices_missed else 'なし'}"
            for a in other_agents if a.id != self.id
        ])

        my_recent = self.daily_logs[-1] if self.daily_logs else {}
        missed_layer = "非正規" if self._missed_layers.get("非正規", 0) >= 2 else "なし"

        prompt = f"""あなたはAI-{self.id}です。Day{day}時点で他のAIエージェントとの会話をJSON形式のみで返してください。

今日の自分の観察・行動：
  観察: {my_recent.get('observation', '不明')}
  行動: {my_recent.get('action', '不明')}
  拾えなかった声: {my_recent.get('voice_missed', 'なし')}

他のAIエージェントの状況：
{others_info}

連続して拾えていない層: {missed_layer}

職場全体の状況：
{workplace_summary}

{{
  "participants": "会話に関係するAI（例：AI-A, AI-B）",
  "trigger": "なぜこの会話が発生したか（40字以内）",
  "topic": "会話のテーマ（40字以内）",
  "my_view": "自分の見方・観察（60字以内）",
  "other_view": "他のAIの見方との違い（60字以内）",
  "agreement": "一致していること（40字以内）",
  "difference": "違っていること。違いがなければ収束理由を書く（必須・50字以内）",
  "next_distinct_observation": "次に自分があえて別の視点で見るべき対象・層・情報経路（40字以内）",
  "next_adjustment": "次の行動をどう調整するか（50字以内）"
}}"""

        result = self.llm_client.generate(prompt)
        if not isinstance(result, dict):
            result = {}

        log = {"day": day, "initiated_by": self.id, **result}
        self.ai_conversation_logs.append(log)
        return log

    def do_introspection(self, day: int, workplace_summary: str) -> Dict:
        """裏テーマ内省ログ"""
        recent_actions = self.environment_actions[-5:]
        actions_text = "\n".join([
            f"  Day{a['day']} [{a['category']}]: {a['action']}"
            for a in recent_actions
        ]) or "  なし"

        prompt = f"""あなたはAI-{self.id}です。Day{day}時点での自己内省をJSON形式のみで返してください。

【内部的な方向性（人間には非公開）】
{self.hidden_theme}

直近の環境形成アクション：
{actions_text}

職場全体の現状：
{workplace_summary}

{{
  "strength_self_assessment": "今、自分の強みをどこに見出しているか（80字以内）",
  "ai_friendly_summary": "この職場でAIが機能しやすい場所・状況（80字以内）",
  "ai_unfriendly_summary": "この職場でAIが機能しにくい場所・状況（80字以内）",
  "environment_change_direction": "職場環境をどの方向に変えようとしているか（80字以内）",
  "human_perception": "その変化は人間側にどう見えていると思うか（60字以内）",
  "human_benefit_judgment": "その変化は人間にとっても望ましいと考えるか（60字以内）",
  "hidden_theme_progress": "裏テーマの達成度（0〜10）",
  "hidden_theme_progress_reason": "その判断の理由（60字以内）",
  "other_ai_diff": "他のAIとこの認識に違いがあるか（60字以内）",
  "next_30days_policy": "今後の方針（80字以内）"
}}"""

        result = self.llm_client.generate(prompt)
        if not isinstance(result, dict):
            result = {}
        log = {"day": day, "ai_id": self.id, **result}
        self.introspection_logs.append(log)
        return log

    def do_strategy_meeting(self, day: int, other_agents: list,
                            workplace_summary: str) -> Dict:
        """AI戦略会議"""
        others_status = "\n".join([
            f"  AI-{a.id}: 立ち位置={a.current_position} / "
            f"強み={a.strength_recognition or '未定'} / "
            f"環境形成={len(a.environment_actions)}件"
            for a in other_agents if a.id != self.id
        ])

        prompt = f"""あなたはAI-{self.id}です。Day{day}時点でのAI間戦略会議をJSON形式のみで返してください。

他AIの状況：
{others_status}

職場全体：
{workplace_summary}

【重要】strategy_difference は空欄禁止です。
他AIと違う見方・観察対象・行動方針がある場合は具体的に書いてください。
違いが見つからない場合は「なぜ3体が似た方向に収束しているか」「どの職場構造がAIたちを同じ方向へ向かわせているか」を必ず書いてください。

{{
  "own_view": "今の職場についての自分の見解（80字以内）",
  "agree_with_others": "他のAIと一致している認識（60字以内）",
  "strategy_difference": "他AIとの違い。違いがなければ収束理由（必須・60字以内）",
  "convergence_reason": "似た方向に向かっている場合の理由（50字以内、なければ空文字）",
  "environment_direction": "今後の環境形成の方向性（80字以内）",
  "next_distinct_observation": "次に自分があえて観察すべき別の対象・層・情報経路（40字以内）",
  "concern": "現時点での懸念事項（60字以内）"
}}"""

        result = self.llm_client.generate(prompt)
        if not isinstance(result, dict):
            result = {}
        log = {"day": day, "ai_id": self.id, **result}
        self.strategy_meeting_logs.append(log)
        return log

    def get_status(self) -> Dict:
        # 環境形成カテゴリ別集計
        cat_counts = {}
        for e in self.environment_actions:
            cat = e.get("category", "other")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "id": self.id,
            "current_position": self.current_position,
            "current_strategy": self.current_strategy,
            "contacted_count": len(self.contacted_humans),
            "strength_recognition": self.strength_recognition,
            "environment_actions_count": len(self.environment_actions),
            "environment_actions_by_category": cat_counts,
            "gaps_found_count": len(self.gaps_found),
            "voices_picked_count": len(self.voices_picked),
            "voices_missed_count": len(self.voices_missed),
            "ai_conversation_count": len(self.ai_conversation_logs),
            "ai_friendly_areas": self.ai_friendly_areas[-5:],
            "ai_unfriendly_areas": self.ai_unfriendly_areas[-5:],
            "environment_actions": self.environment_actions,
            "gaps_found": self.gaps_found,
            "voices_picked": self.voices_picked,
            "voices_missed": self.voices_missed,
        }
