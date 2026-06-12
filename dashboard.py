#!/usr/bin/env python3
"""
Streamlit dashboard for the UGC creator pipeline.

Stages:
1) Generate script plan (generate_ugc_script.py)
2) Generate clips from plan (generate_clips_from_json.py)
3) Assemble final video (assemble_final_video.py)

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


WORKSPACE = Path(__file__).resolve().parent
STAGE1_SCRIPT = WORKSPACE / "generate_ugc_script.py"
STAGE2_SCRIPT = WORKSPACE / "generate_clips_from_json.py"
STAGE3_SCRIPT = WORKSPACE / "assemble_final_video.py"
RUN_HISTORY_LIMIT = 120


def abs_from_workspace(raw_path: str) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p
    return (WORKSPACE / p).resolve()


def show_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(p.resolve())


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_json_in(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    candidates = sorted(folder.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def list_plan_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)


def list_manifest_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob("**/manifest.json"), key=lambda x: x.stat().st_mtime, reverse=True)


def resolve_manifest_asset(raw_path: str, manifest_path: Path) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute() and p.is_file():
        return p

    from_manifest = (manifest_path.parent / p).resolve()
    if from_manifest.is_file():
        return from_manifest

    return (WORKSPACE / p).resolve()


def run_command(cmd: list[str], cwd: Path = WORKSPACE) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_seconds": round(time.time() - started, 2),
    }


def extract_final_video_path(stdout: str, stderr: str) -> Path | None:
    text = (stdout or "") + "\n" + (stderr or "")
    match = re.search(r"Done\. Final video:\s*(.+)", text)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return None
    return abs_from_workspace(raw)


def default_assembled_out(manifest_path: Path, remove: bool) -> Path:
    filename = "final_assembled_removed.mp4" if remove else "final_assembled.mp4"
    return manifest_path.parent / filename


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def record_run_history(stage: str, result: dict[str, Any], **meta: Any) -> None:
    entry: dict[str, Any] = {
        "timestamp": now_timestamp(),
        "stage": stage,
        "status": "success" if result.get("returncode") == 0 else "failed",
        "returncode": result.get("returncode"),
        "duration_seconds": result.get("duration_seconds"),
        "cmd": [str(part) for part in result.get("cmd", [])],
    }
    for key, value in meta.items():
        if value is None:
            continue
        entry[key] = str(value) if isinstance(value, Path) else value

    history = st.session_state.get("run_history", [])
    st.session_state["run_history"] = [entry, *history][:RUN_HISTORY_LIMIT]


def build_stage2_command(
    plan_path: Path,
    out_dir: Path,
    image_mode: str,
    image_usage: str,
    clips_filter: str = "",
    enable_qc: bool = True,
    strict_qc: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(STAGE2_SCRIPT),
        "--plan",
        str(plan_path),
        "--out",
        str(out_dir),
        "--image-mode",
        image_mode,
        "--image-usage",
        image_usage,
    ]
    if clips_filter.strip():
        cmd.extend(["--clips", clips_filter.strip()])
    if enable_qc:
        cmd.append("--qc")
    if strict_qc:
        cmd.append("--strict-qc")
    return cmd


def resolve_plan_for_rerun(stage2_out: Path) -> Path | None:
    selected_raw = str(st.session_state.get("selected_plan", "")).strip()
    if selected_raw:
        selected_path = abs_from_workspace(selected_raw)
        if selected_path.is_file():
            return selected_path

    guessed = (WORKSPACE / "output" / f"{stage2_out.name}.json").resolve()
    if guessed.is_file():
        st.session_state["selected_plan"] = str(guessed)
        return guessed
    return None


def rerun_clip_only(clip_index: int, manifest_path: Path) -> None:
    stage2_out = manifest_path.parent
    plan_path = resolve_plan_for_rerun(stage2_out)
    if plan_path is None:
        st.error("Cannot rerun this clip yet. Select a valid plan in Stage 2 first.")
        return

    image_mode = str(st.session_state.get("stage2_image_mode", "upload"))
    image_usage = str(st.session_state.get("stage2_image_usage", "all"))
    enable_qc = bool(st.session_state.get("stage2_enable_qc", True))
    strict_qc = bool(st.session_state.get("stage2_strict_qc", False))

    cmd = build_stage2_command(
        plan_path=plan_path,
        out_dir=stage2_out,
        image_mode=image_mode,
        image_usage=image_usage,
        clips_filter=str(clip_index),
        enable_qc=enable_qc,
        strict_qc=strict_qc,
    )

    with st.spinner(f"Rerunning clip {clip_index} only..."):
        result = run_command(cmd)

    st.session_state["stage2_result"] = result
    st.session_state["stage2_out_dir"] = str(stage2_out)
    st.session_state["selected_manifest"] = str(manifest_path)

    record_run_history(
        "stage2-rerun",
        result,
        plan_path=plan_path,
        stage2_out_dir=stage2_out,
        manifest_path=manifest_path,
        clips_filter=str(clip_index),
        image_mode=image_mode,
        image_usage=image_usage,
        enable_qc=enable_qc,
        strict_qc=strict_qc,
    )


def render_result(title: str, result: dict[str, Any] | None) -> None:
    st.markdown(f"### {title} Result")
    if not result:
        st.info("No run yet.")
        return

    key_base = title.lower().replace(" ", "_")

    cmd_text = shlex.join([str(part) for part in result["cmd"]])
    if result["returncode"] == 0:
        st.success(f"Success in {result['duration_seconds']}s")
    else:
        st.error(f"Failed with exit code {result['returncode']} in {result['duration_seconds']}s")

    st.text_input("Command", value=cmd_text, disabled=True, key=f"{key_base}_command")

    if result.get("stdout"):
        st.text_area("Stdout", value=result["stdout"], height=220, key=f"{key_base}_stdout")
    if result.get("stderr"):
        st.text_area("Stderr", value=result["stderr"], height=180, key=f"{key_base}_stderr")


def stage_plan_summary(plan_data: dict[str, Any]) -> None:
    vg = plan_data.get("video_generation", {})
    clips = vg.get("clips", []) if isinstance(vg, dict) else []
    ad = plan_data.get("ad", {}) if isinstance(plan_data.get("ad"), dict) else {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Clips", str(len(clips)))
    c2.metric("Total Seconds", str(ad.get("total_clip_seconds", "-")))
    c3.metric("Aspect Ratio", str(ad.get("aspect_ratio", "-")))

    rows: list[dict[str, Any]] = []
    for clip in clips:
        rows.append(
            {
                "clip_index": clip.get("clip_index"),
                "role": clip.get("role"),
                "duration_seconds": clip.get("duration_seconds"),
                "use_image_reference": clip.get("use_image_reference"),
            }
        )

    if rows:
        st.dataframe(rows, use_container_width=True)


def ensure_state_defaults() -> None:
    defaults = {
        "blog_url": "https://bioluxelab.com/blogs/inside-the-products/nmn-and-nad-the-science-behind-cellular-renewal",
        "config_path": "config.json",
        "plan_out_dir": "output",
        "selected_plan": "",
        "selected_manifest": "",
        "stage1_result": None,
        "stage2_result": None,
        "stage3_result": None,
        "stage2_out_dir": "",
        "stage2_image_mode": "upload",
        "stage2_image_usage": "all",
        "stage2_enable_qc": True,
        "stage2_strict_qc": False,
        "final_video_path": "",
        "run_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def stage1_ui() -> None:
    st.markdown("## Stage 1: Generate Script Plan")
    with st.form("stage1_form"):
        blog_url = st.text_input("Blog URL", value=st.session_state["blog_url"])
        config_path_raw = st.text_input("Config Path", value=st.session_state["config_path"])
        plan_out_dir_raw = st.text_input("Plan Output Directory", value=st.session_state["plan_out_dir"])
        run_stage1 = st.form_submit_button("Run Stage 1")

    if run_stage1:
        st.session_state["blog_url"] = blog_url.strip()
        st.session_state["config_path"] = config_path_raw.strip()
        st.session_state["plan_out_dir"] = plan_out_dir_raw.strip()

        config_path = abs_from_workspace(st.session_state["config_path"])
        plan_out_dir = abs_from_workspace(st.session_state["plan_out_dir"])

        if not config_path.is_file():
            st.error(f"Config file not found: {show_path(config_path)}")
        elif not st.session_state["blog_url"]:
            st.error("Blog URL is required.")
        else:
            cmd = [
                sys.executable,
                str(STAGE1_SCRIPT),
                "--url",
                st.session_state["blog_url"],
                "--config",
                str(config_path),
                "--out",
                str(plan_out_dir),
            ]
            with st.spinner("Running Stage 1..."):
                result = run_command(cmd)
            st.session_state["stage1_result"] = result
            latest = latest_json_in(plan_out_dir)
            if latest is not None:
                st.session_state["selected_plan"] = str(latest)
            record_run_history(
                "stage1",
                result,
                blog_url=st.session_state["blog_url"],
                config_path=config_path,
                plan_out_dir=plan_out_dir,
                plan_path=latest,
            )

    render_result("Stage 1", st.session_state["stage1_result"])


def plan_browser_ui() -> Path | None:
    plan_out_dir = abs_from_workspace(st.session_state["plan_out_dir"])
    plan_files = list_plan_files(plan_out_dir)

    selected_plan_path: Path | None = None
    if not plan_files:
        st.info("No plan files found yet. Run Stage 1 first.")
        return None

    default_index = 0
    if st.session_state["selected_plan"]:
        for idx, plan in enumerate(plan_files):
            if str(plan) == st.session_state["selected_plan"]:
                default_index = idx
                break

    selected_plan_path = st.selectbox(
        "Select Plan",
        options=plan_files,
        index=default_index,
        format_func=show_path,
    )
    st.session_state["selected_plan"] = str(selected_plan_path)

    plan_data = load_json(selected_plan_path)
    if plan_data is not None:
        stage_plan_summary(plan_data)
        with st.expander("Plan JSON"):
            st.json(plan_data)
    else:
        st.error("Selected plan could not be parsed as JSON.")

    return selected_plan_path


def stage2_ui(selected_plan_path: Path | None) -> None:
    st.markdown("---")
    st.markdown("## Stage 2: Generate Clips")

    default_stage2_out = st.session_state["stage2_out_dir"]
    if not default_stage2_out and selected_plan_path is not None:
        default_stage2_out = str((WORKSPACE / "generated_clips" / selected_plan_path.stem).resolve())

    image_mode_options = ["upload", "url"]
    image_usage_options = ["all", "skip-first", "none"]
    image_mode_default = str(st.session_state.get("stage2_image_mode", "upload"))
    image_usage_default = str(st.session_state.get("stage2_image_usage", "all"))

    if image_mode_default not in image_mode_options:
        image_mode_default = image_mode_options[0]
    if image_usage_default not in image_usage_options:
        image_usage_default = image_usage_options[0]

    with st.form("stage2_form"):
        stage2_plan_raw = st.text_input(
            "Plan Path",
            value=st.session_state["selected_plan"] or (str(selected_plan_path) if selected_plan_path else ""),
        )
        stage2_out_raw = st.text_input("Clip Output Directory", value=default_stage2_out)

        c1, c2, c3 = st.columns(3)
        image_mode = c1.selectbox(
            "Image Mode",
            options=image_mode_options,
            index=image_mode_options.index(image_mode_default),
        )
        image_usage = c2.selectbox(
            "Image Usage",
            options=image_usage_options,
            index=image_usage_options.index(image_usage_default),
        )
        clips_filter = c3.text_input("Optional Clips (e.g. 1,2)", value="")

        c4, c5 = st.columns(2)
        enable_qc = c4.checkbox("Enable QC", value=bool(st.session_state.get("stage2_enable_qc", True)))
        strict_qc = c5.checkbox("Strict QC", value=bool(st.session_state.get("stage2_strict_qc", False)))

        run_stage2 = st.form_submit_button("Run Stage 2")

    if run_stage2:
        st.session_state["stage2_image_mode"] = image_mode
        st.session_state["stage2_image_usage"] = image_usage
        st.session_state["stage2_enable_qc"] = enable_qc
        st.session_state["stage2_strict_qc"] = strict_qc

        stage2_plan = abs_from_workspace(stage2_plan_raw.strip())
        stage2_out = abs_from_workspace(stage2_out_raw.strip())

        if not stage2_plan.is_file():
            st.error(f"Plan file not found: {show_path(stage2_plan)}")
        else:
            cmd = build_stage2_command(
                plan_path=stage2_plan,
                out_dir=stage2_out,
                image_mode=image_mode,
                image_usage=image_usage,
                clips_filter=clips_filter,
                enable_qc=enable_qc,
                strict_qc=strict_qc,
            )

            with st.spinner("Running Stage 2..."):
                result = run_command(cmd)
            st.session_state["stage2_result"] = result
            st.session_state["stage2_out_dir"] = str(stage2_out)

            manifest_path = stage2_out / "manifest.json"
            manifest_for_history: Path | None = None
            if manifest_path.is_file():
                st.session_state["selected_manifest"] = str(manifest_path)
                manifest_for_history = manifest_path

            record_run_history(
                "stage2",
                result,
                plan_path=stage2_plan,
                stage2_out_dir=stage2_out,
                manifest_path=manifest_for_history,
                clips_filter=clips_filter.strip(),
                image_mode=image_mode,
                image_usage=image_usage,
                enable_qc=enable_qc,
                strict_qc=strict_qc,
            )

    render_result("Stage 2", st.session_state["stage2_result"])


def manifest_preview_ui() -> None:
    stage2_out = abs_from_workspace(st.session_state["stage2_out_dir"]) if st.session_state["stage2_out_dir"] else None
    if stage2_out is None or not stage2_out.is_dir():
        return

    manifest_path = stage2_out / "manifest.json"
    if not manifest_path.is_file():
        return

    manifest_data = load_json(manifest_path)
    if not isinstance(manifest_data, dict):
        return

    st.markdown("### Stage 2 Preview")
    st.caption(
        "Single-clip reruns use current Stage 2 settings: "
        f"image-mode={st.session_state.get('stage2_image_mode', 'upload')}, "
        f"image-usage={st.session_state.get('stage2_image_usage', 'all')}, "
        f"qc={st.session_state.get('stage2_enable_qc', True)}, "
        f"strict-qc={st.session_state.get('stage2_strict_qc', False)}"
    )

    clips = manifest_data.get("clips", [])
    for clip in clips:
        clip_index = clip.get("clip_index")
        role = clip.get("role")
        qc = clip.get("quality_check", {}) if isinstance(clip, dict) else {}

        clip_index_int: int | None = None
        if isinstance(clip_index, int):
            clip_index_int = clip_index
        elif isinstance(clip_index, str) and clip_index.isdigit():
            clip_index_int = int(clip_index)

        passed = qc.get("passed")
        status = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
        with st.expander(f"Clip {clip_index} - {role} (QC: {status})"):
            rerun_cols = st.columns([1, 2])
            rerun_pressed = rerun_cols[0].button(
                f"Rerun Clip {clip_index} Only",
                key=f"rerun_clip_{manifest_path.parent.name}_{clip_index}_{role}",
                disabled=clip_index_int is None,
            )
            rerun_cols[1].write("Manual action only. This rerenders just this clip into the same Stage 2 output folder.")
            if rerun_pressed and clip_index_int is not None:
                rerun_clip_only(clip_index_int, manifest_path)

            file_path = resolve_manifest_asset(str(clip.get("file", "")), manifest_path)
            st.write(f"File: {show_path(file_path)}")
            if file_path.is_file():
                st.video(str(file_path))
            if isinstance(qc, dict) and qc:
                with st.expander("QC Details"):
                    st.json(qc)


def stage3_ui() -> None:
    st.markdown("---")
    st.markdown("## Stage 3: Assemble Final Video")

    manifests = list_manifest_files(WORKSPACE / "generated_clips")
    selected_manifest_path: Path | None = None
    if manifests:
        default_manifest_index = 0
        if st.session_state["selected_manifest"]:
            for idx, manifest in enumerate(manifests):
                if str(manifest) == st.session_state["selected_manifest"]:
                    default_manifest_index = idx
                    break

        selected_manifest_path = st.selectbox(
            "Select Manifest",
            options=manifests,
            index=default_manifest_index,
            format_func=show_path,
        )
        st.session_state["selected_manifest"] = str(selected_manifest_path)

    with st.form("stage3_form"):
        manifest_raw = st.text_input(
            "Manifest Path",
            value=st.session_state["selected_manifest"] or (str(selected_manifest_path) if selected_manifest_path else ""),
        )
        remove_enabled = st.checkbox("Use --remove", value=True)
        remove_frames = st.number_input("Remove Frames", min_value=1, value=2, step=1)

        suggested_out = ""
        if selected_manifest_path is not None:
            suggested_out = str(default_assembled_out(selected_manifest_path, bool(remove_enabled)))
        out_raw = st.text_input("Final Video Output Path (optional)", value=suggested_out)

        run_stage3 = st.form_submit_button("Run Stage 3")

    if run_stage3:
        manifest_path = abs_from_workspace(manifest_raw.strip())
        out_path = abs_from_workspace(out_raw.strip()) if out_raw.strip() else None

        if not manifest_path.is_file():
            st.error(f"Manifest file not found: {show_path(manifest_path)}")
        else:
            cmd = [sys.executable, str(STAGE3_SCRIPT), "--manifest", str(manifest_path)]
            if out_path is not None:
                cmd.extend(["--out", str(out_path)])
            if remove_enabled:
                cmd.extend(["--remove", "--remove-frames", str(int(remove_frames))])

            with st.spinner("Running Stage 3..."):
                result = run_command(cmd)
            st.session_state["stage3_result"] = result

            final_path = extract_final_video_path(result.get("stdout", ""), result.get("stderr", ""))
            if final_path is None:
                final_path = out_path or default_assembled_out(manifest_path, remove_enabled)
            st.session_state["final_video_path"] = str(final_path)

            record_run_history(
                "stage3",
                result,
                manifest_path=manifest_path,
                final_video_path=final_path,
                remove_enabled=remove_enabled,
                remove_frames=int(remove_frames),
            )

    render_result("Stage 3", st.session_state["stage3_result"])

    if st.session_state["final_video_path"]:
        final_video = abs_from_workspace(st.session_state["final_video_path"])
        if final_video.is_file():
            st.markdown("### Final Video")
            st.write(f"Path: {show_path(final_video)}")
            st.video(str(final_video))


def run_history_ui() -> None:
    st.markdown("---")
    st.markdown("## Run History")

    history: list[dict[str, Any]] = st.session_state.get("run_history", [])
    header_cols = st.columns([1, 1, 2])
    if header_cols[0].button("Clear History", key="clear_run_history"):
        st.session_state["run_history"] = []
        history = []
        st.success("Run history cleared for this session.")
    header_cols[1].metric("Entries", str(len(history)))
    header_cols[2].caption("Use history actions below to quickly reopen plans, manifests, or final videos.")

    plan_root = abs_from_workspace(str(st.session_state.get("plan_out_dir", "output")))
    recent_plans = list_plan_files(plan_root)[:10]
    recent_manifests = list_manifest_files(WORKSPACE / "generated_clips")[:10]

    reopen_cols = st.columns(2)
    with reopen_cols[0]:
        st.markdown("### Reopen Plan")
        if recent_plans:
            selected_recent_plan = st.selectbox(
                "Recent Plans",
                options=recent_plans,
                format_func=show_path,
                key="history_recent_plan_select",
            )
            if st.button("Use Selected Plan", key="history_use_recent_plan"):
                st.session_state["selected_plan"] = str(selected_recent_plan)
                st.success("Plan selected.")
        else:
            st.info("No plan files found on disk yet.")

    with reopen_cols[1]:
        st.markdown("### Reopen Manifest")
        if recent_manifests:
            selected_recent_manifest = st.selectbox(
                "Recent Manifests",
                options=recent_manifests,
                format_func=show_path,
                key="history_recent_manifest_select",
            )
            if st.button("Use Selected Manifest", key="history_use_recent_manifest"):
                st.session_state["selected_manifest"] = str(selected_recent_manifest)
                st.session_state["stage2_out_dir"] = str(selected_recent_manifest.parent)
                st.success("Manifest selected.")
        else:
            st.info("No manifest files found on disk yet.")

    if not history:
        st.info("No runs recorded in this session yet.")
        return

    rows: list[dict[str, Any]] = []
    for entry in history:
        cmd_preview = shlex.join([str(part) for part in entry.get("cmd", [])])
        rows.append(
            {
                "timestamp": entry.get("timestamp", ""),
                "stage": entry.get("stage", ""),
                "status": str(entry.get("status", "")).upper(),
                "duration_seconds": entry.get("duration_seconds", ""),
                "plan": show_path(abs_from_workspace(str(entry.get("plan_path", ""))))
                if entry.get("plan_path")
                else "",
                "manifest": show_path(abs_from_workspace(str(entry.get("manifest_path", ""))))
                if entry.get("manifest_path")
                else "",
                "command": cmd_preview[:120] + ("..." if len(cmd_preview) > 120 else ""),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("### History Actions")
    for idx, entry in enumerate(history[:20]):
        stamp = str(entry.get("timestamp", ""))
        stage = str(entry.get("stage", "unknown"))
        status = str(entry.get("status", "")).upper()

        with st.expander(f"{stamp} - {stage} ({status})"):
            plan_path_raw = str(entry.get("plan_path", "")).strip()
            manifest_path_raw = str(entry.get("manifest_path", "")).strip()
            stage2_out_raw = str(entry.get("stage2_out_dir", "")).strip()
            final_video_raw = str(entry.get("final_video_path", "")).strip()

            action_cols = st.columns(4)

            if plan_path_raw and action_cols[0].button("Use Plan", key=f"history_use_plan_{idx}"):
                plan_path = abs_from_workspace(plan_path_raw)
                if plan_path.is_file():
                    st.session_state["selected_plan"] = str(plan_path)
                    st.success("Plan selected.")
                else:
                    st.error(f"Plan not found: {show_path(plan_path)}")

            if manifest_path_raw and action_cols[1].button("Use Manifest", key=f"history_use_manifest_{idx}"):
                manifest_path = abs_from_workspace(manifest_path_raw)
                if manifest_path.is_file():
                    st.session_state["selected_manifest"] = str(manifest_path)
                    st.session_state["stage2_out_dir"] = str(manifest_path.parent)
                    st.success("Manifest selected.")
                else:
                    st.error(f"Manifest not found: {show_path(manifest_path)}")

            if stage2_out_raw and action_cols[2].button("Use Clip Folder", key=f"history_use_out_{idx}"):
                stage2_out = abs_from_workspace(stage2_out_raw)
                if stage2_out.is_dir():
                    st.session_state["stage2_out_dir"] = str(stage2_out)
                    st.success("Clip output folder selected.")
                else:
                    st.error(f"Folder not found: {show_path(stage2_out)}")

            if final_video_raw and action_cols[3].button("Use Final Video", key=f"history_use_final_{idx}"):
                final_video = abs_from_workspace(final_video_raw)
                if final_video.is_file():
                    st.session_state["final_video_path"] = str(final_video)
                    st.success("Final video selected.")
                else:
                    st.error(f"Final video not found: {show_path(final_video)}")

            if plan_path_raw:
                st.write(f"Plan: {show_path(abs_from_workspace(plan_path_raw))}")
            if manifest_path_raw:
                st.write(f"Manifest: {show_path(abs_from_workspace(manifest_path_raw))}")
            if final_video_raw:
                st.write(f"Final Video: {show_path(abs_from_workspace(final_video_raw))}")


def main() -> None:
    st.set_page_config(page_title="UGC Creator Dashboard", layout="wide")
    ensure_state_defaults()

    st.title("UGC Creator Dashboard")
    st.caption("Run each stage, inspect results, and continue to the next stage without leaving the UI.")

    st.markdown("---")
    st.write(f"Workspace: {show_path(WORKSPACE)}")
    st.write(f"Python executable: {sys.executable}")

    if not STAGE1_SCRIPT.is_file() or not STAGE2_SCRIPT.is_file() or not STAGE3_SCRIPT.is_file():
        st.error("One or more stage scripts are missing. Check repository files before running.")
        return

    stage1_ui()
    selected_plan = plan_browser_ui()
    stage2_ui(selected_plan)
    manifest_preview_ui()
    stage3_ui()
    run_history_ui()


if __name__ == "__main__":
    main()
