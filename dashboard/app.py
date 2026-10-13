import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VisualGuard AI", page_icon="◉", layout="wide")
API = os.environ.get("VG_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = str(exc)
        if getattr(exc, "response", None) is not None:
            try:
                detail = exc.response.json().get("detail", detail)
            except ValueError:
                pass
        st.error(f"API request failed: {detail}")
        return None


st.markdown(
    """<style>
.stApp {background:#0b1220;color:#e7edf6}
[data-testid="stSidebar"] {background:#111d2e}
[data-testid="stMetric"] {background:#152237;border:1px solid #27364b;
border-radius:12px;padding:18px}
.eyebrow {font-size:12px;letter-spacing:3px;color:#58cbbb;margin-bottom:8px}
.subtitle {color:#a7b6c9;max-width:780px;margin-bottom:28px}
</style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("VisualGuard")
    st.caption("WORKPLACE SAFETY ANALYTICS")
    page = st.radio("Workspace", ["Analyze & review", "Event log", "Analytics", "Model & data"])
    st.divider()
    health = api("GET", "/health")
    if health:
        st.success("API connected")
        st.caption(f"Backend: {health['backend']} · v{health['version']}")
    st.caption("Local review prototype. Each alert needs a person's assessment.")

st.markdown('<div class="eyebrow">VISUALGUARD AI / REVIEW WORKSPACE</div>', unsafe_allow_html=True)

if page == "Analyze & review":
    st.title("See the evidence behind each alert")
    st.markdown(
        '<div class="subtitle">Inspect PPE observations, follow people across a clip, '
        "and review the moments that need attention.</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Start with a sample")
            st.write(
                "The generated 8-second scene exercises detection fixtures, tracking, "
                "video jobs, and review. It contains no real workers or model predictions."
            )
        with right:
            if st.button("Run synthetic demo", type="primary", use_container_width=True):
                result = api("POST", "/api/demo")
                if result:
                    st.session_state["job_id"] = result["id"]
                    st.success("Demo queued. The video worker will process it.")
    with st.expander("Analyze your own image or recorded video"):
        st.warning(
            "Experimental detector: it can miss uncovered heads. "
            "No alert does not mean a person is wearing the required PPE."
        )
        st.caption(
            "Uploads run locally. Use Model & data to inspect the checkpoint's measured limits."
        )
        upload = st.file_uploader("Choose media", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        if st.button("Analyze upload", disabled=upload is None):
            endpoint = (
                "images" if upload.name.lower().endswith(("jpg", "jpeg", "png")) else "videos"
            )
            result = api(
                "POST", f"/api/{endpoint}", files={"file": (upload.name, upload.getvalue())}
            )
            if result:
                st.session_state["job_id"] = result["id"]
    jobs = api("GET", "/api/jobs") or []
    if jobs:
        ids = [job["id"] for job in jobs]
        current = st.session_state.get("job_id", ids[0])
        selected = st.selectbox(
            "Analysis history",
            ids,
            index=ids.index(current) if current in ids else 0,
            format_func=lambda key: next(
                f"{j['kind'].title()} · {'Synthetic' if j['synthetic'] == 'yes' else 'Uploaded'}"
                f" · {j['status']} · {key[:8]}"
                for j in jobs
                if j["id"] == key
            ),
        )
        if st.button("Refresh job status"):
            st.rerun()
        job = api("GET", f"/api/jobs/{selected}")
        if job:
            if job["status"] in {"queued", "running"}:
                st.progress(
                    job["progress"] or 0, text=f"{job['status'].title()} — refresh for updates"
                )
                st.caption("If this stays queued, start the worker using scripts/start.ps1.")
            elif job["status"] == "failed":
                st.error(job["error"])
            else:
                result = job["result"]
                if result["synthetic"]:
                    st.warning(
                        "SYNTHETIC DEMO · workflow evidence only; not ML accuracy or throughput."
                    )
                else:
                    st.warning(
                        "EXPERIMENTAL MODEL · predictions need review; "
                        "video event accuracy has not been established."
                    )
                a, b, c = st.columns(3)
                a.metric("Candidate events", len(result["events"]))
                b.metric("Frames inspected", result["sampled_frames"])
                c.metric("Processing time", f"{result['processing_seconds']:.2f} s")
                media_url = f"{API}/api/jobs/{selected}/media/{result['artifact']}"
                try:
                    media = requests.get(media_url, timeout=60)
                    media.raise_for_status()
                    if result["kind"] == "video":
                        st.video(media.content)
                    else:
                        st.image(media.content)
                except requests.RequestException as exc:
                    st.error(f"Could not load annotated media: {exc}")
                st.subheader("Timeline")
                chart = pd.DataFrame(
                    [
                        {
                            "Seconds": row["timestamp"],
                            "People": row["people"],
                            "Possible missing helmet": sum(
                                o["helmet"] == "possible_missing" for o in row["observations"]
                            ),
                            "Helmet unknown": sum(
                                o["helmet"] == "unknown" for o in row["observations"]
                            ),
                        }
                        for row in result["timeline"]
                    ]
                ).set_index("Seconds")
                st.line_chart(chart)
                st.download_button(
                    "Download evidence JSON",
                    data=__import__("json").dumps(result, indent=2),
                    file_name=f"visualguard-{selected[:8]}.json",
                    mime="application/json",
                )
                with st.expander("Inspect boxes, confidence, and rules"):
                    st.json(result)
    else:
        st.info("No analyses yet. Run the synthetic demo to inspect the full workflow.")

elif page == "Event log":
    st.title("Review candidate events")
    st.write(
        "A detection becomes an event only after consistent evidence. Record your assessment here."
    )
    status = st.selectbox(
        "Review status", ["all", "pending", "confirmed", "false_alert", "uncertain"]
    )
    job_index = {job["id"]: job for job in (api("GET", "/api/jobs") or [])}
    job_filter = st.selectbox(
        "Analysis",
        ["all", *job_index],
        format_func=lambda key: (
            "All analyses"
            if key == "all"
            else f"{key[:8]} · {job_index[key]['kind']} · "
            f"{'Synthetic' if job_index[key]['synthetic'] == 'yes' else 'Uploaded'}"
        ),
    )
    params = {} if status == "all" else {"review": status}
    if job_filter != "all":
        params["job_id"] = job_filter
    rows = api("GET", "/api/events", params=params) or []
    if not rows:
        st.info("No events match this filter.")
    for event in rows:
        payload = event["payload"]
        synthetic = event["synthetic"] == "yes"
        label = "SYNTHETIC · " if synthetic else ""
        with st.expander(
            f"{label}Helmet review · person #{payload['track_id']} · "
            f"{payload['start_seconds']:.2f}–{payload['end_seconds']:.2f}s · {event['review']}"
            f" · {event['job_id'][:8]}"
        ):
            st.write(payload["rule"])
            st.write(payload["evidence"]["reason"])
            st.caption(
                f"Job {event['job_id']} · evidence frames {payload['evidence_frames']} · "
                f"minimum detection confidence {payload['confidence']:.2f} (not event probability)"
            )
            st.json(payload["evidence"], expanded=False)
            choices = ["pending", "confirmed", "false_alert", "uncertain"]
            with st.form(event["id"]):
                review = st.selectbox("Assessment", choices, index=choices.index(event["review"]))
                note = st.text_area("Review note", event["note"], max_chars=2000)
                if st.form_submit_button("Save review"):
                    result = api(
                        "PATCH",
                        f"/api/events/{event['id']}/review",
                        json={"status": review, "note": note},
                    )
                    if result:
                        st.rerun()

elif page == "Analytics":
    st.title("Understand the observed activity")
    synthetic = st.toggle("Show synthetic demo analyses", value=True)
    if synthetic:
        st.warning("Synthetic fixtures only. These counts do not measure workplace safety.")
    data = api("GET", "/api/analytics", params={"synthetic": str(synthetic).lower()})
    if data:
        a, b, c, d = st.columns(4)
        a.metric("Completed videos", data["completed_videos"])
        b.metric("Candidate events", data["candidate_events"])
        c.metric("Observed person-minutes", f"{data['observed_person_minutes']:.2f}")
        rate = data["candidate_events_per_person_minute"]
        d.metric("Events / person-minute", "—" if rate is None else f"{rate:.2f}")
        st.caption(data["denominator"])
        fraction = data["unknown_helmet_fraction"]
        st.write(
            "Helmet visibility unknown: "
            + ("no observations" if fraction is None else f"{fraction:.1%}")
        )
        if data["review_counts"]:
            st.subheader("Review outcomes")
            st.bar_chart(pd.Series(data["review_counts"], name="Events"))
        if data["events_by_video_minute"]:
            st.subheader("Event starts by minute within the source clips")
            st.bar_chart(pd.Series(data["events_by_video_minute"], name="Events"))

else:
    st.title("Model and data status")
    model = api("GET", "/api/model")
    if model and model.get("report"):
        report = model["report"]
        st.warning("Experimental CPU baseline. Not validated for workplace safety decisions.")
        st.write(report["summary"])
        metrics = report["evaluation"]
        a, b, c = st.columns(3)
        split_label = "Validation images" if metrics["split"] == "val" else "Test images"
        a.metric(split_label, metrics["images"])
        for column, label, key in (
            (b, "mAP50", "metrics/mAP50(B)"),
            (c, "mAP50–95", "metrics/mAP50-95(B)"),
        ):
            value = metrics["metrics"].get(key)
            column.metric(label, "Not measured" if value is None else f"{value:.1%}")
        st.caption(
            "mAP summarizes detection across classes and thresholds; it is not safety accuracy."
        )
        if metrics.get("head_profile"):
            st.caption(
                "AP was not recomputed after head box refinement. "
                "The table below evaluates the actual app detections."
            )
        if metrics["split"] == "val":
            st.info(
                "These validation images were used during model development. "
                "An independent final evaluation is still needed."
            )
        if not model.get("evaluation_matches_configuration", False):
            st.warning(
                "The current inference settings differ from this evaluation. "
                "These recorded metrics do not describe the current settings."
            )
        st.subheader("At the evaluated confidence threshold")
        st.dataframe(pd.DataFrame(metrics["fixed_operating_point"]["per_class"]).T)
        point = metrics["fixed_operating_point"]
        st.caption(
            f"One-to-one detection matching at IoU {metrics['fixed_operating_point']['iou']} "
            f"and confidence {metrics['fixed_operating_point']['confidence']}. "
            f"Head confidence: {point.get('head_confidence', point['confidence'])}. "
            f"Input size: {metrics['imgsz']} pixels. "
            "Counts are objects, not people or events."
        )
        for limitation in report["limitations"]:
            st.write(f"• {limitation}")
        with st.expander("Checkpoint and evaluation evidence"):
            st.json(report)
        if experiment := report.get("latest_experiment"):
            st.subheader("Latest training experiment")
            st.write(experiment["summary"])
            comparison = experiment["comparison"]
            st.caption(
                f"Development comparison on {comparison['images']} validation images, "
                f"{comparison['imgsz']}-pixel input, confidence {comparison['confidence']}. "
                "These images were used during development; this is not independent test evidence."
            )
            comparison_rows = []
            for name, previous in comparison["baseline_per_class"].items():
                candidate = comparison["candidate_per_class"][name]
                comparison_rows.append(
                    {
                        "Class": name,
                        "Baseline precision": previous["precision"],
                        "Candidate precision": candidate["precision"],
                        "Baseline recall": previous["recall"],
                        "Candidate recall": candidate["recall"],
                    }
                )
            st.dataframe(pd.DataFrame(comparison_rows).set_index("Class"))
            with st.expander("Candidate selection checks"):
                st.dataframe(pd.DataFrame(comparison["gates"]))
    else:
        st.info("No matching evaluation report is available for the configured checkpoint.")
    st.write(
        "The demo uses colored geometry generated locally. The YOLO adapter accepts local weights "
        "with person, head, helmet, and vest labels. "
        "General COCO weights cannot replace a PPE model."
    )
    st.subheader("What still needs to be established")
    st.markdown(
        "- An independent test set with verified source scenes\n"
        "- Better uncovered-head detection across workplace conditions\n"
        "- Manually reviewed clips for tracking and event evaluation"
    )
    st.subheader("Known limits")
    st.write(
        "Vest absence remains unknown. The baseline tracker can switch IDs. Detection confidence "
        "is not the probability of an unsafe situation. Camera zones, Grad-CAM, anomaly models, "
        "and the optional VLM assistant are future work."
    )
