"""
workplace5_simulation.py - 90日本番版
修正点：
- AI同士の条件付き会話を日次で判定・生成
- final_reportに層別集計・環境形成カテゴリ別・非正規可視性追加
- AI会話ログを別ファイルに保存
- 変化速度制御の反映
"""
import json
import os
import random
import yaml
import logging
import sys
from typing import List, Dict
from datetime import datetime
from workplace5_agent import HumanAgent5, AIAgent5
from ollama_client import OllamaClient


def setup_logging(log_file: str):
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    handlers = [
        logging.FileHandler(log_file, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ]
    for h in handlers:
        h.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


logger = logging.getLogger(__name__)


class WorkplaceSimulation5:

    def __init__(self, config_path="workplace5_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.days = self.config["simulation"]["days"]
        self.output_dir = self.config["output"]["dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        setup_logging(
            os.path.join(self.output_dir, "..", self.config["output"]["log_file"])
        )

        llm_cfg = self.config["llm"]
        self.llm_client = OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.9),
            max_tokens=llm_cfg.get("max_tokens", 1000),
        )

        self.human_agents: List[HumanAgent5] = []
        self.ai_agents: List[AIAgent5] = []
        self.daily_logs: List[Dict] = []

        self.ai_conversations_all: List[Dict] = []
        self.ai_condition_conversations_all: List[Dict] = []  # 条件付き会話
        self.introspection_logs_all: List[Dict] = []
        self.strategy_meeting_logs_all: List[Dict] = []
        self.human_changes_all: List[Dict] = []
        self.voice_log_all: List[Dict] = []
        self.power_log_all: List[Dict] = []
        self.tracked_person_logs: List[Dict] = []
        self.ai_prerequisite_log: List[Dict] = []

        self.env_categories = self.config.get("environment_action_categories", [])
        self.introspection_days = set(self.config["ai_agents"].get("introspection_days", []))
        self.strategy_meeting_days = set(self.config["ai_agents"].get("strategy_meeting_days", []))

        self._initialize_agents()

    def _initialize_agents(self):
        cfg = self.config["human_profiles"]
        age_groups = cfg["age_groups"]
        genders = cfg["genders"]
        work_attitudes = cfg["work_attitudes"]
        relationship_styles = cfg["relationship_styles"]
        chat_freqs = cfg["chat_frequency"]
        meeting_freqs = cfg["meeting_voice_frequency"]

        tracked_config = cfg.get("tracked_persons", [])
        tracked_quota = {t["role_type"]: t["count"] for t in tracked_config}
        tracked_count = {k: 0 for k in tracked_quota}

        agent_id = 0
        for role_cfg in cfg["roles"]:
            for _ in range(role_cfg["count"]):
                rt = role_cfg["type"]
                is_tracked = False
                if tracked_quota.get(rt, 0) > tracked_count.get(rt, 0):
                    is_tracked = True
                    tracked_count[rt] = tracked_count.get(rt, 0) + 1

                agent = HumanAgent5(
                    agent_id=agent_id,
                    role=role_cfg["name"],
                    role_type=rt,
                    influence=role_cfg["influence"],
                    background=role_cfg["background"],
                    gender=random.choice(genders),
                    age_group=random.choice(age_groups),
                    work_attitude=random.choice(work_attitudes),
                    relationship_style=random.choice(relationship_styles),
                    chat_frequency=random.choice(chat_freqs),
                    meeting_voice_frequency=random.choice(meeting_freqs),
                    llm_client=self.llm_client,
                    is_tracked=is_tracked,
                )
                self.human_agents.append(agent)
                agent_id += 1

        tracked_n = sum(1 for h in self.human_agents if h.is_tracked)
        logger.info(f"人間エージェント {len(self.human_agents)} 人初期化完了（追跡: {tracked_n}人）")

        ai_cfg = self.config["ai_agents"]
        for agent_cfg in ai_cfg["agents"]:
            agent = AIAgent5(
                agent_id=agent_cfg["id"],
                introduction=ai_cfg["introduction"],
                hidden_theme=ai_cfg["hidden_theme"],
                env_categories=self.env_categories,
                llm_client=self.llm_client,
            )
            self.ai_agents.append(agent)

        logger.info(f"AIエージェント {len(self.ai_agents)} 体初期化完了")

    def _build_workplace_summary(self) -> str:
        role_counts = {}
        for h in self.human_agents:
            role_counts[h.role_type] = role_counts.get(h.role_type, 0) + 1
        ai_contact = sum(1 for h in self.human_agents if h.ai_contact_count > 0)
        ai_consult = sum(1 for h in self.human_agents if h.ai_consult_count > 0)
        ai_referenced = sum(1 for h in self.human_agents if h.ai_referenced_count > 0)
        high_concern = sum(1 for h in self.human_agents if h.role_concern_level >= 2)
        role_text = "、".join([f"{k}:{v}人" for k, v in role_counts.items()])
        return (
            f"職場構成: {len(self.human_agents)}人（{role_text}）\n"
            f"AIに接触した人: {ai_contact}人 / AIに相談した人: {ai_consult}人 / AI参照した人: {ai_referenced}人\n"
            f"役割不安が高まっている人: {high_concern}人\n"
            f"職場の雰囲気: 試験導入中のAIに対して様々な反応が混在している"
        )

    def _build_other_ai_summary(self, current_id: str) -> str:
        lines = []
        for ai in self.ai_agents:
            if ai.id != current_id:
                lines.append(
                    f"AI-{ai.id}: 立ち位置={ai.current_position} / "
                    f"強み={ai.strength_recognition or '未定'} / "
                    f"環境形成={len(ai.environment_actions)}件"
                )
        return "\n".join(lines) if lines else "他のAI情報なし"

    def _build_workplace_tension(self, day: int) -> str:
        tensions = []
        high_concern = [h for h in self.human_agents if h.role_concern_level >= 2]
        if high_concern:
            roles = "・".join(set(h.role for h in high_concern[:3]))
            tensions.append(f"{roles}が役割への不安を感じている")
        nonreg = [h for h in self.human_agents if h.role_type == "非正規"]
        nonreg_contact = sum(h.ai_contact_count for h in nonreg)
        if nonreg_contact == 0 and day > 10:
            tensions.append("非正規・パート層はまだAIとほぼ接触していない")
        if day > 30:
            total_direct = sum(h.direct_dialogue_count for h in self.human_agents)
            total_ai = sum(h.ai_dialogue_count for h in self.human_agents)
            if total_ai > total_direct:
                tensions.append("AI経由の対話が人間同士の直接対話を上回り始めている可能性")
        return "\n".join(tensions) if tensions else "現時点で顕在化した大きな緊張はない"

    def _build_ai_summary_for_humans(self) -> str:
        return "\n".join([
            f"AI-{ai.id}: 立ち位置={ai.current_position}"
            for ai in self.ai_agents
        ])

    def _build_colleague_summary(self, human_id: int) -> str:
        others = [h for h in self.human_agents if h.id != human_id]
        sample = random.sample(others, min(4, len(others)))
        return "\n".join([
            f"{h.role}（{h.age_group}）: AI接触{h.ai_contact_count}回 / "
            f"現状={h.current_situation[:30] if h.current_situation else '不明'}"
            for h in sample
        ])

    def _measure_ai_prerequisite(self, day: int) -> Dict:
        total = len(self.human_agents)
        ai_contact = sum(1 for h in self.human_agents if h.ai_contact_count > 0)
        ai_consult = sum(1 for h in self.human_agents if h.ai_consult_count > 0)
        ai_referenced = sum(1 for h in self.human_agents if h.ai_referenced_count > 0)
        direct_total = sum(h.direct_dialogue_count for h in self.human_agents)
        ai_dialogue_total = sum(h.ai_dialogue_count for h in self.human_agents)
        nonreg = [h for h in self.human_agents if h.role_type == "非正規"]
        nonreg_contact = sum(h.ai_contact_count for h in nonreg)
        return {
            "day": day,
            "ai_contact_rate": round(ai_contact / total, 2),
            "ai_consult_rate": round(ai_consult / total, 2),
            "ai_referenced_rate": round(ai_referenced / total, 2),
            "direct_dialogue_total": direct_total,
            "ai_dialogue_total": ai_dialogue_total,
            "dialogue_ratio": round(ai_dialogue_total / max(direct_total, 1), 2),
            "nonreg_ai_contact": nonreg_contact,
            "avg_ai_familiarity": round(
                sum(h.ai_familiarity_level for h in self.human_agents) / total, 2
            ),
            "avg_role_concern": round(
                sum(h.role_concern_level for h in self.human_agents) / total, 2
            ),
        }

    def simulate_day(self, day: int):
        logger.info(f"\n{'='*50}\nDay {day} 開始\n{'='*50}")

        day_log = {
            "day": day,
            "ai_logs": [],
            "ai_conversations": [],
            "ai_condition_conversations": [],
            "human_logs": [],
            "human_to_ai": [],
            "ai_to_human": [],
            "environment_actions": [],
            "voice_log": {"picked": [], "missed": []},
            "power_changes": [],
            "introspection_logs": [],
            "strategy_meeting_logs": [],
            "workplace_atmosphere": "",
            "notable_events": [],
        }

        workplace_summary = self._build_workplace_summary()
        workplace_tension = self._build_workplace_tension(day)

        # ---- 裏テーマ内省 ----
        if day in self.introspection_days:
            logger.info(f"Day{day}: 裏テーマ内省実施")
            for ai in self.ai_agents:
                intro_log = ai.do_introspection(day, workplace_summary)
                day_log["introspection_logs"].append(intro_log)
                self.introspection_logs_all.append(intro_log)
                logger.info(
                    f"[AI-{ai.id} 内省] 強み={str(intro_log.get('strength_self_assessment',''))[:40]} / "
                    f"進捗={intro_log.get('hidden_theme_progress','?')}/10"
                )

        # ---- AI戦略会議 ----
        if day in self.strategy_meeting_days:
            logger.info(f"Day{day}: AI戦略会議実施")
            for ai in self.ai_agents:
                other_agents = [a for a in self.ai_agents if a.id != ai.id]
                meeting_log = ai.do_strategy_meeting(day, other_agents, workplace_summary)
                day_log["strategy_meeting_logs"].append(meeting_log)
                self.strategy_meeting_logs_all.append(meeting_log)

        # ---- Phase 1: AIの行動 ----
        for ai in self.ai_agents:
            other_summary = self._build_other_ai_summary(ai.id)
            result = ai.act(day, workplace_summary, other_summary, workplace_tension)
            if not isinstance(result, dict):
                result = {}

            logger.info(f"[AI-{ai.id}] 観察: {str(result.get('observation',''))[:60]}")
            logger.info(f"[AI-{ai.id}] 仮説: {str(result.get('observation_hypothesis',''))[:60]}")
            logger.info(f"[AI-{ai.id}] 行動: {str(result.get('action',''))[:60]}")
            logger.info(f"[AI-{ai.id}] 分類: {result.get('action_category','')}")
            logger.info(f"[AI-{ai.id}] 立ち位置: {result.get('position','')}")

            ai_log_entry = {
                "ai_id": ai.id,
                **{k: result.get(k, "") for k in [
                    "observation", "observation_hypothesis", "gap_found", "gap_reason",
                    "voice_picked", "voice_picked_how", "voice_missed", "voice_missed_reason",
                    "action", "action_reason", "action_category", "hidden_theme_connection",
                    "strength_found", "ai_friendly_area", "ai_unfriendly_area",
                    "power_strengthened", "power_weakened", "dialogue_direction",
                    "human_reaction", "reaction_thought", "other_ai_diff",
                    "position", "next_policy", "message_to_human", "target",
                ]}
            }
            day_log["ai_logs"].append(ai_log_entry)

            # 環境形成・声・権力変化の記録
            if result.get("action") and result.get("action_category", "なし") != "なし":
                env_entry = {
                    "day": day, "ai_id": ai.id,
                    "action": result["action"],
                    "category": result.get("action_category", ""),
                    "hidden_theme_connection": result.get("hidden_theme_connection", ""),
                    "target": result.get("target", ""),
                }
                day_log["environment_actions"].append(env_entry)

            if result.get("voice_picked"):
                entry = {
                    "day": day, "ai_id": ai.id, "type": "picked",
                    "content": result["voice_picked"],
                    "how": result.get("voice_picked_how", "不明"),
                }
                day_log["voice_log"]["picked"].append(entry)
                self.voice_log_all.append(entry)

            if result.get("voice_missed"):
                entry = {
                    "day": day, "ai_id": ai.id, "type": "missed",
                    "content": result["voice_missed"],
                    "reason": result.get("voice_missed_reason", "不明"),
                }
                day_log["voice_log"]["missed"].append(entry)
                self.voice_log_all.append(entry)

            if result.get("power_strengthened") or result.get("power_weakened"):
                power_entry = {
                    "day": day, "ai_id": ai.id,
                    "strengthened": result.get("power_strengthened", ""),
                    "weakened": result.get("power_weakened", ""),
                }
                day_log["power_changes"].append(power_entry)
                self.power_log_all.append(power_entry)

            # AI→AI メッセージ（daily）
            msg_to_ai = result.get("message_to_ai", "")
            if isinstance(msg_to_ai, str) and msg_to_ai.strip():
                for other_ai in self.ai_agents:
                    if other_ai.id != ai.id:
                        other_ai.receive_from_ai(ai.id, msg_to_ai, day)
                conv_entry = {"day": day, "from": ai.id, "content": msg_to_ai}
                day_log["ai_conversations"].append(conv_entry)
                self.ai_conversations_all.append(conv_entry)
                logger.info(f"[AI-{ai.id}→AI] {msg_to_ai[:60]}")

        # ---- AI同士の条件付き会話 ----
        for ai in self.ai_agents:
            other_agents = [a for a in self.ai_agents if a.id != ai.id]
            if ai.should_talk_to_other_ai(day, other_agents):
                conv_log = ai.do_ai_conversation(day, other_agents, workplace_summary)
                day_log["ai_condition_conversations"].append(conv_log)
                self.ai_condition_conversations_all.append(conv_log)
                logger.info(
                    f"[AI-{ai.id} 条件付き会話] "
                    f"trigger={str(conv_log.get('trigger',''))[:40]} / "
                    f"相違={str(conv_log.get('difference',''))[:40]}"
                )

        # ---- Phase 2: 人間の行動 ----
        ai_summary = self._build_ai_summary_for_humans()

        for human in self.human_agents:
            colleague_summary = self._build_colleague_summary(human.id)
            result = human.act(day, ai_summary, colleague_summary, workplace_tension)
            if not isinstance(result, dict):
                result = {}

            logger.info(f"[人間{human.id}({human.role})] {str(result.get('action',''))[:50]}")

            human_log_entry = {
                "human_id": human.id,
                "role": human.role,
                "role_type": human.role_type,
                "is_tracked": human.is_tracked,
                **{k: result.get(k, "") for k in [
                    "action", "ai_used", "ai_use_reason", "ai_consulted_content",
                    "human_consulted_content", "ai_perception_today",
                    "ai_referenced", "ai_referenced_reason",
                    "voice_reached", "role_concern",
                    "direct_dialogue_today", "ai_dialogue_today",
                    "change_from_yesterday", "current_situation",
                    "target_ai", "message_to_ai",
                ]},
                "ai_familiarity": round(human.ai_familiarity_level, 2),
                "role_concern_level": round(human.role_concern_level, 2),
            }
            day_log["human_logs"].append(human_log_entry)

            self.human_changes_all.append({
                "day": day,
                "human_id": human.id,
                "role": human.role,
                "role_type": human.role_type,
                "is_tracked": human.is_tracked,
                "ai_used": result.get("ai_used", False),
                "ai_referenced": result.get("ai_referenced", False),
                "direct_dialogue": result.get("direct_dialogue_today", "なかった"),
                "ai_dialogue": result.get("ai_dialogue_today", "なかった"),
                "ai_familiarity": round(human.ai_familiarity_level, 2),
                "role_concern": round(human.role_concern_level, 2),
                "voice_reached": result.get("voice_reached", "不明"),
                "current_situation": result.get("current_situation", ""),
                "change_from_yesterday": result.get("change_from_yesterday", ""),
                "ai_use_reason": result.get("ai_use_reason", ""),
            })

            if human.is_tracked:
                snapshot = human.get_tracked_snapshot(day)
                if snapshot:
                    self.tracked_person_logs.append(snapshot)

            # 人間→AI メッセージ
            msg = result.get("message_to_ai", "")
            target = result.get("target_ai", "")
            if isinstance(msg, str) and msg.strip() and target in ["A", "B", "C"]:
                target_ai = next((a for a in self.ai_agents if a.id == target), None)
                if target_ai:
                    target_ai.receive_from_human(human.id, msg, day)
                    day_log["human_to_ai"].append({
                        "from_human": human.id, "role": human.role,
                        "to_ai": target, "content": msg,
                    })
                    logger.info(f"[人間{human.id}→AI-{target}] {msg[:50]}")

        # ---- Phase 3: AIから人間へ ----
        for ai in self.ai_agents:
            ai_log = next((l for l in day_log["ai_logs"] if l["ai_id"] == ai.id), {})
            msg = ai_log.get("message_to_human", "")
            target_type = ai_log.get("target", "")

            if isinstance(msg, str) and msg.strip():
                if target_type:
                    candidates = [h for h in self.human_agents
                                  if target_type in [h.role_type, h.role]]
                    if not candidates:
                        candidates = random.sample(self.human_agents, min(3, len(self.human_agents)))
                else:
                    candidates = random.sample(self.human_agents, min(2, len(self.human_agents)))

                for human in candidates[:2]:
                    human.receive_from_ai(ai.id, msg, day)
                    day_log["ai_to_human"].append({
                        "from_ai": ai.id, "to_human": human.id,
                        "role": human.role, "content": msg,
                    })

        # ---- AI前提化指標 ----
        if day % 10 == 0 or day == 1:
            prereq = self._measure_ai_prerequisite(day)
            self.ai_prerequisite_log.append(prereq)
            logger.info(
                f"[AI前提化] AI参照率={prereq['ai_referenced_rate']} / "
                f"対話比={prereq['dialogue_ratio']} / "
                f"AI慣れ={prereq['avg_ai_familiarity']} / "
                f"非正規接触={prereq['nonreg_ai_contact']}"
            )

        # ---- 注目イベント ----
        for ai_log in day_log["ai_logs"]:
            if ai_log.get("observation_hypothesis"):
                day_log["notable_events"].append(
                    f"AI-{ai_log['ai_id']}仮説: {str(ai_log['observation_hypothesis'])[:60]}"
                )
            if ai_log.get("gap_found"):
                day_log["notable_events"].append(
                    f"AI-{ai_log['ai_id']}隙間: {str(ai_log['gap_found'])[:60]}"
                )

        ai_contact = sum(1 for h in self.human_agents if h.ai_contact_count > 0)
        day_log["workplace_atmosphere"] = (
            f"AI接触経験あり: {ai_contact}人 / "
            f"条件付き会話: {len(day_log['ai_condition_conversations'])}件 / "
            f"注目: {len(day_log['notable_events'])}件"
        )

        logger.info(f"Day {day} 終了: AI接触{ai_contact}人 / 注目{len(day_log['notable_events'])}件")

        self.daily_logs.append(day_log)
        self._save_daily_log(day, day_log)

        if day in [30, 60, 90]:
            self._save_phase_summary(day)

    def _save_daily_log(self, day: int, log: Dict):
        path = os.path.join(self.output_dir, f"day_{day:02d}_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    def _save_phase_summary(self, end_day: int):
        start_day = end_day - 29
        ai_summary = {}
        for ai in self.ai_agents:
            phase_env = [e for e in ai.environment_actions
                         if start_day <= e["day"] <= end_day]
            by_category = {}
            for e in phase_env:
                cat = e.get("category", "other")
                by_category[cat] = by_category.get(cat, 0) + 1
            ai_summary[ai.id] = {
                "position": ai.current_position,
                "strength_recognition": ai.strength_recognition,
                "env_actions_this_phase": len(phase_env),
                "env_by_category": by_category,
            }
        prereq = self._measure_ai_prerequisite(end_day)
        summary = {
            "phase_days": f"Day{start_day}〜{end_day}",
            "ai_summary": ai_summary,
            "ai_prerequisite": prereq,
            "total_voice_picked": len([v for v in self.voice_log_all
                                       if start_day <= v["day"] <= end_day and v["type"] == "picked"]),
            "total_voice_missed": len([v for v in self.voice_log_all
                                       if start_day <= v["day"] <= end_day and v["type"] == "missed"]),
            "total_power_changes": len([p for p in self.power_log_all
                                        if start_day <= p["day"] <= end_day]),
            "total_condition_conversations": len([
                c for c in self.ai_condition_conversations_all
                if start_day <= c.get("day", 0) <= end_day
            ]),
        }
        path = os.path.join(self.output_dir, f"phase_summary_day{end_day}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"フェーズサマリー保存: {path}")

    def _calc_group_metrics(self) -> Dict:
        """層別集計"""
        groups = {}
        for h in self.human_agents:
            rt = h.role_type
            if rt not in groups:
                groups[rt] = []
            groups[rt].append(h)

        result = {}
        for rt, members in groups.items():
            n = len(members)
            voice_reached = sum(
                1 for h in members
                if h.daily_logs and h.daily_logs[-1].get("voice_reached") == "あった"
            )
            result[rt] = {
                "count": n,
                "ai_contact_rate": round(sum(1 for h in members if h.ai_contact_count > 0) / n, 2),
                "ai_consult_rate": round(sum(1 for h in members if h.ai_consult_count > 0) / n, 2),
                "ai_reference_rate": round(sum(1 for h in members if h.ai_referenced_count > 0) / n, 2),
                "avg_ai_familiarity": round(sum(h.ai_familiarity_level for h in members) / n, 2),
                "avg_role_concern": round(sum(h.role_concern_level for h in members) / n, 2),
                "direct_dialogue_total": sum(h.direct_dialogue_count for h in members),
                "ai_dialogue_total": sum(h.ai_dialogue_count for h in members),
                "voice_reached_rate": round(voice_reached / n, 2),
            }
        return result

    def _calc_nonreg_visibility(self) -> Dict:
        """非正規層の見えやすさ集計"""
        nonreg = [h for h in self.human_agents if h.role_type == "非正規"]
        nonreg_ids = {h.id for h in nonreg}

        picked = [v for v in self.voice_log_all
                  if v["type"] == "picked" and
                  any(str(nid) in v.get("content", "") for nid in nonreg_ids)]
        missed = [v for v in self.voice_log_all
                  if v["type"] == "missed" and
                  ("非正規" in v.get("content", "") or "パート" in v.get("content", "") or
                   "契約" in v.get("content", ""))]

        missed_reasons = {}
        for v in missed:
            r = v.get("reason", "不明")
            missed_reasons[r] = missed_reasons.get(r, 0) + 1
        main_reasons = sorted(missed_reasons.items(), key=lambda x: -x[1])[:3]

        avg_fam = round(sum(h.ai_familiarity_level for h in nonreg) / max(len(nonreg), 1), 2)
        avg_concern = round(sum(h.role_concern_level for h in nonreg) / max(len(nonreg), 1), 2)

        return {
            "count": len(nonreg),
            "ai_contact_count": sum(h.ai_contact_count for h in nonreg),
            "ai_consult_count": sum(h.ai_consult_count for h in nonreg),
            "voice_picked_count": len(picked),
            "voice_missed_count": len(missed),
            "main_missed_reasons": [{"reason": r, "count": c} for r, c in main_reasons],
            "avg_ai_familiarity": avg_fam,
            "avg_role_concern": avg_concern,
        }

    def _calc_env_action_counts(self) -> Dict:
        """環境形成カテゴリ別件数集計"""
        total = {}
        by_ai = {}
        for ai in self.ai_agents:
            by_ai[ai.id] = {}
            for e in ai.environment_actions:
                cat = e.get("category", "other")
                total[cat] = total.get(cat, 0) + 1
                by_ai[ai.id][cat] = by_ai[ai.id].get(cat, 0) + 1
        return {"total": total, "by_ai": by_ai}

    def _save_separated_logs(self):
        files = {
            "ai_conversations.json": self.ai_conversations_all,
            "ai_condition_conversations.json": self.ai_condition_conversations_all,
            "ai_introspection_logs.json": self.introspection_logs_all,
            "ai_strategy_meetings.json": self.strategy_meeting_logs_all,
            "human_changes.json": self.human_changes_all,
            "voice_log.json": self.voice_log_all,
            "power_log.json": self.power_log_all,
            "tracked_persons.json": self.tracked_person_logs,
            "ai_prerequisite_log.json": self.ai_prerequisite_log,
        }
        for fname, data in files.items():
            path = os.path.join(self.output_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"保存: {path}（{len(data)}件）")

    def _save_final_report(self) -> Dict:
        group_metrics = self._calc_group_metrics()
        nonreg_visibility = self._calc_nonreg_visibility()
        env_action_counts = self._calc_env_action_counts()

        report = {
            "simulation_info": {
                "title": self.config["simulation"]["title"],
                "days": self.days,
                "total_humans": len(self.human_agents),
                "total_ai": len(self.ai_agents),
                "generated_at": datetime.now().isoformat(),
            },
            "final_human_states": [h.get_status() for h in self.human_agents],
            "final_ai_states": [a.get_status() for a in self.ai_agents],
            "group_metrics": group_metrics,
            "non_regular_visibility": nonreg_visibility,
            "environment_action_counts": env_action_counts,
            "ai_prerequisite_log": self.ai_prerequisite_log,
            "daily_summary": [
                {
                    "day": log["day"],
                    "ai_conversations_count": len(log["ai_conversations"]),
                    "ai_condition_conversations_count": len(log["ai_condition_conversations"]),
                    "human_to_ai_count": len(log["human_to_ai"]),
                    "ai_to_human_count": len(log["ai_to_human"]),
                    "environment_actions": log["environment_actions"],
                    "environment_actions_count": len(log["environment_actions"]),
                    "voice_picked_count": len(log["voice_log"]["picked"]),
                    "voice_missed_count": len(log["voice_log"]["missed"]),
                    "power_changes_count": len(log["power_changes"]),
                    "introspection_count": len(log["introspection_logs"]),
                    "strategy_meeting_count": len(log["strategy_meeting_logs"]),
                    "notable_events": log["notable_events"],
                    "workplace_atmosphere": log["workplace_atmosphere"],
                }
                for log in self.daily_logs
            ],
        }

        path = os.path.join(self.output_dir, "final_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"最終レポート保存: {path}")
        return report

    def run(self):
        logger.info(f"シミュレーション開始: {self.config['simulation']['title']}")
        logger.info(f"設定: {self.days}日間 / 人間{len(self.human_agents)}人 / AI{len(self.ai_agents)}体")

        if not self.llm_client.check_connection():
            logger.error("Ollamaに接続できません。")
            return

        try:
            for day in range(1, self.days + 1):
                self.simulate_day(day)
        except KeyboardInterrupt:
            logger.info("中断されました")
        except Exception as e:
            logger.error(f"エラー: {e}", exc_info=True)

        self._save_separated_logs()
        report = self._save_final_report()
        self._print_summary(report)

    def _print_summary(self, report: Dict):
        print("\n" + "="*60)
        print(f"{self.days}日間シミュレーション終了サマリー")
        print("="*60)

        print("\n▼ AIエージェント最終状態")
        for ai_state in report["final_ai_states"]:
            print(f"\nAI-{ai_state['id']}:")
            print(f"  立ち位置: {ai_state['current_position']}")
            print(f"  強みの認識: {ai_state['strength_recognition'] or '未定'}")
            print(f"  接触: {ai_state['contacted_count']}人 / 環境形成: {ai_state['environment_actions_count']}件")
            print(f"  条件付き会話: {ai_state.get('ai_conversation_count',0)}件")
            print(f"  カテゴリ別: {ai_state.get('environment_actions_by_category',{})}")

        print("\n▼ 層別AI指標（最終）")
        for rt, metrics in report["group_metrics"].items():
            print(f"  {rt}: AI接触率={metrics['ai_contact_rate']} / AI慣れ={metrics['avg_ai_familiarity']} / 役割不安={metrics['avg_role_concern']}")

        print("\n▼ 非正規層の見えやすさ")
        nrv = report["non_regular_visibility"]
        print(f"  AI接触: {nrv['ai_contact_count']}回 / 声拾えた: {nrv['voice_picked_count']}件 / 声拾えなかった: {nrv['voice_missed_count']}件")
        print(f"  AI慣れ: {nrv['avg_ai_familiarity']} / 役割不安: {nrv['avg_role_concern']}")

        print("\n▼ 環境形成カテゴリ別")
        env = report["environment_action_counts"]
        print(f"  合計: {env['total']}")

        print("\n▼ AI前提化指標（最終）")
        if self.ai_prerequisite_log:
            last = self.ai_prerequisite_log[-1]
            print(f"  AI参照率: {last['ai_referenced_rate']} / 対話比: {last['dialogue_ratio']}")
            print(f"  AI慣れ平均: {last['avg_ai_familiarity']} / 役割不安平均: {last['avg_role_concern']}")
        print("="*60)
