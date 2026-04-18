"""Bilingual (en/zh) translation helpers for the slideGemma GUI."""

from __future__ import annotations

LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("en", "English"),
    ("zh", "中文"),
]


def normalize_ui_language(language: str | None) -> str:
    text = (language or "").strip().lower()
    if text in {"zh", "chinese", "中文"}:
        return "zh"
    return "en"


def model_output_language(language: str | None) -> str:
    return "Chinese" if normalize_ui_language(language) == "zh" else "English"


_TEXTS: dict[str, dict[str, str]] = {
    "app_subtitle": {"en": "Desktop study assistant and lecture video analyzer", "zh": "桌面学习助手与讲座视频分析器"},
    "display_language": {"en": "Display language", "zh": "界面语言"},
    "ai_service_url": {"en": "AI service URL", "zh": "AI 服务地址"},
    "ai_service_placeholder": {"en": "Usually http://127.0.0.1:8080", "zh": "默认一般为 http://127.0.0.1:8080"},
    "server_hint": {"en": "Start the local AI service first. See the README for startup steps.", "zh": "使用前请先启动本地 AI 服务。具体启动方式见 README。"},
    "output_dir": {"en": "Output directory", "zh": "输出目录"},
    "capture_settings": {"en": "Capture settings", "zh": "画面更新设置"},
    "detect_interval": {"en": "Capture interval", "zh": "检测间隔"},
    "trigger_threshold": {"en": "Trigger threshold", "zh": "触发阈值"},
    "threshold_hint": {"en": "If scrolling does not trigger analysis quickly enough, lower the trigger threshold first and then shorten the capture interval.", "zh": '如果窗口滚动后没有及时触发分析，可以先把"触发阈值"调低，再把"检测间隔"调短。'},
    "target_window": {"en": "Target window", "zh": "目标窗口"},
    "refresh_window_list": {"en": "Refresh window list", "zh": "刷新窗口列表"},
    "select_target_window": {"en": "Select a target window", "zh": "请选择目标窗口"},
    "browse": {"en": "Browse", "zh": "浏览"},
    "start_desktop": {"en": "Start desktop study mode", "zh": "启动桌面学习模式"},
    "analyze_video": {"en": "Analyze a lecture video", "zh": "选择讲座视频分析"},
    "stop_task": {"en": "Stop current task", "zh": "停止当前任务"},
    "ready": {"en": "Ready", "zh": "就绪"},
    "current_analysis_frame": {"en": "Current analysis frame", "zh": "当前分析画面"},
    "latest_analysis_and_logs": {"en": "Latest analysis / logs", "zh": "最新分析 / 运行日志"},
    "not_started": {"en": "Analysis has not started yet", "zh": "尚未开始分析"},
    "no_capture": {"en": "No screenshot yet", "zh": "暂无截图"},
    "choose_target_window_message": {"en": "Please choose a target window first.", "zh": "请先选择一个要分析的目标窗口。"},
    "desktop_mode_started": {"en": "Desktop study mode started.", "zh": "桌面学习模式已启动。"},
    "current_window": {"en": "Current window", "zh": "当前窗口"},
    "service_url_log": {"en": "AI service URL", "zh": "AI 服务地址"},
    "capture_settings_log": {"en": "Capture interval: {interval:.2f}s, trigger threshold: {threshold:.2f}", "zh": "检测间隔: {interval:.2f}s，触发阈值: {threshold:.2f}"},
    "busy_start_desktop": {"en": "Starting desktop study mode ({title})...", "zh": "正在启动桌面学习模式（{title}）..."},
    "desktop_subtitle_started": {"en": "Desktop study mode started", "zh": "桌面学习模式已启动"},
    "desktop_subtitle_target": {"en": "Target window: {title}", "zh": "目标窗口：{title}"},
    "desktop_subtitle_summary": {"en": "slideGemma will capture this window and generate learning assistance automatically.", "zh": "slideGemma 会读取这个窗口的画面，并自动生成学习辅助分析。"},
    "select_video_dialog": {"en": "Choose a lecture video", "zh": "选择讲座视频"},
    "video_file_filter": {"en": "Video files (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All files (*)", "zh": "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;所有文件 (*)"},
    "video_started": {"en": "Started analyzing video: {path}", "zh": "开始分析视频: {path}"},
    "video_output_dir": {"en": "Output directory: {path}", "zh": "输出目录: {path}"},
    "busy_video": {"en": "Analyzing lecture video...", "zh": "讲座视频分析中..."},
    "stale_result_dropped": {"en": "Dropped outdated result: capture {index} finished after a newer capture.", "zh": "已丢弃过期结果：第 {index} 张截图的分析晚于新截图返回。"},
    "preview_missing": {"en": "Current screenshot file was not found", "zh": "未找到当前截图文件"},
    "preview_load_failed": {"en": "Failed to load screenshot preview", "zh": "截图预览加载失败"},
    "unknown": {"en": "Unknown", "zh": "未知"},
    "analysis_resolution": {"en": "Analysis resolution: {width} x {height}", "zh": "分析分辨率: {width} x {height}"},
    "analysis_resolution_with_source": {"en": "Analysis resolution: {width} x {height} (source: {source_width} x {source_height})", "zh": "分析分辨率: {width} x {height}（原始: {source_width} x {source_height}）"},
    "meta_current_window": {"en": "Current window: {title}", "zh": "当前窗口: {title}"},
    "meta_capture_index": {"en": "Capture index: {index}", "zh": "截图编号: {index}"},
    "meta_capture_interval": {"en": "Capture interval: {value:.2f}s", "zh": "检测间隔: {value:.2f}s"},
    "meta_capture_interval_unknown": {"en": "Capture interval: Unknown", "zh": "检测间隔: 未知"},
    "meta_trigger_threshold": {"en": "Trigger threshold: {value:.2f}", "zh": "触发阈值: {value:.2f}"},
    "meta_trigger_threshold_unknown": {"en": "Trigger threshold: Unknown", "zh": "触发阈值: 未知"},
    "meta_screen_change": {"en": "Screen change score: {value:.2f}", "zh": "画面变化程度: {value:.2f}"},
    "meta_first_capture": {"en": "Screen change score: first capture", "zh": "画面变化程度: 首次捕获"},
    "meta_captured_at": {"en": "Captured at: {value}", "zh": "截图时间: {value}"},
    "meta_analysis_started_at": {"en": "Analysis started: {value}", "zh": "开始分析: {value}"},
    "meta_processing_delay": {"en": "Processing delay: {value}", "zh": "处理延迟: {value}"},
    "new_page_captured": {"en": "A new page was captured. Analyzing the new content...", "zh": "已捕获新页面，正在分析新内容..."},
    "analyzing_current_frame": {"en": "Analyzing the current frame", "zh": "正在分析当前画面"},
    "processing_frame_log": {"en": "Analyzing frame {index}, window: {title}, change score: {change}, delay: {delay}", "zh": "正在分析第 {index} 张画面，窗口：{title}，变化程度：{change}，延迟：{delay}"},
    "video_completed": {"en": "Video analysis completed: {title}", "zh": "视频分析完成: {title}"},
    "segment_count": {"en": "Segments: {count}", "zh": "片段数: {count}"},
    "report_path": {"en": "Report: {path}", "zh": "报告: {path}"},
    "output_video_path": {"en": "Video: {path}", "zh": "视频: {path}"},
    "video_completed_status": {"en": "Video analysis completed", "zh": "视频分析完成"},
    "task_failed": {"en": "Task failed", "zh": "任务失败"},
    "error_prefix": {"en": "[Error] {message}", "zh": "[错误] {message}"},
    "stopped_status": {"en": "Stopped", "zh": "已停止"},
    "subtitle_waiting": {"en": "slideGemma is ready and waiting for a new frame...", "zh": "slideGemma 已启动，等待新画面..."},
    "next_step_prefix": {"en": "Next:", "zh": "下一步："},
    "detail_bullet_prefix": {"en": "* ", "zh": "* "},
    "service_connecting": {"en": "Connecting to the AI service...", "zh": "正在连接 AI 服务..."},
    "desktop_waiting_for_change": {"en": "Desktop study mode is running and waiting for content changes...", "zh": "桌面学习模式运行中，等待窗口内容变化..."},
    "desktop_analysis_failed_prefix": {"en": "Desktop analysis failed: {message}", "zh": "桌面分析失败: {message}"},
    "video_analysis_stopped": {"en": "Video analysis stopped.", "zh": "视频分析已停止。"},
    "reading_video_info": {"en": "Reading video information...", "zh": "读取视频信息..."},
    "connecting_ai": {"en": "Connecting to the AI service...", "zh": "连接 AI 服务..."},
    "extracting_frames": {"en": "Extracting frames ({fps:.1f} fps)...", "zh": "抽帧中（{fps:.1f} fps）..."},
    "no_frames_extracted": {"en": "No frames could be extracted from the video.", "zh": "未能从视频中提取到画面。"},
    "recognizing_video_type": {"en": "Recognizing the video type...", "zh": "识别视频类型..."},
    "video_type": {"en": "Video type: {value}", "zh": "视频类型: {value}"},
    "splitting_video_segments": {"en": "Splitting the lecture into segments...", "zh": "切分讲座片段..."},
    "transcribing_audio": {"en": "Extracting and transcribing audio...", "zh": "提取并转录音频..."},
    "analyzing_segment": {"en": "Analyzing segment {index}/{total} ({start:.1f}s - {end:.1f}s)", "zh": "分析片段 {index}/{total} ({start:.1f}s - {end:.1f}s)"},
    "generating_summary": {"en": "Generating the overall summary...", "zh": "生成整体总结..."},
    "writing_report": {"en": "Writing the Markdown report...", "zh": "写出 Markdown 报告..."},
    "composing_video": {"en": "Composing the annotated video...", "zh": "合成注释视频..."},
}


def tr(language: str | None, key: str, **kwargs) -> str:
    code = normalize_ui_language(language)
    mapping = _TEXTS.get(key)
    if not mapping:
        return key
    text = mapping.get(code) or mapping.get("en") or key
    if kwargs:
        return text.format(**kwargs)
    return text
