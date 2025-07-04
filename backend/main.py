from features.ai_brain import AIBrain
from features.web_search import web_search
from features.autotrade import run_autotrader
from features.self_reflect import SelfReflection
from features.self_audit import SelfAudit
from features.dashboard import TerminalDashboard
from features.conversation_manager import conversation_manager
from features.knowledge_base import knowledge_base
from core.plugin_manager import plugin_manager
from utils.logging_config import setup_logging, get_logger
from database.config import init_database
from config.settings import settings

import subprocess
import threading
import time
from pathlib import Path

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def main():
    logger.info("JARVIS v3.0 starting up", extra={"version": settings.APP_VERSION})
    
    # Initialize database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"❌ Database initialization failed: {e}")
        return
    
    # Initialize AI components
    brain = AIBrain()
    
    # Load plugins
    try:
        plugin_results = plugin_manager.load_all_plugins()
        loaded_plugins = sum(plugin_results.values())
        logger.info(f"Loaded {loaded_plugins} plugins")
        if loaded_plugins > 0:
            print(f"🔌 Loaded {loaded_plugins} plugins")
    except Exception as e:
        logger.warning(f"Plugin loading failed: {e}")
    
    # Initialize knowledge base
    try:
        knowledge_base.load_from_disk()
        if len(knowledge_base.documents) == 0:
            # Populate from existing Q&A data
            populated = knowledge_base.populate_from_qa_history(500)
            if populated > 0:
                knowledge_base.save_to_disk()
                print(f"📚 Populated knowledge base with {populated} entries")
        else:
            print(f"📚 Knowledge base loaded ({len(knowledge_base.documents)} documents)")
    except Exception as e:
        logger.warning(f"Knowledge base initialization failed: {e}")
    
    # Start conversation session
    session_id = conversation_manager.start_new_session("interactive")
    
    print(f"🤖 {settings.APP_NAME} is online. Type 'exit' to quit.")
    print(f"💬 Session: {session_id}")
    
    online_mode = settings.WEB_SEARCH_ENABLED
    logger.info("JARVIS initialization complete", extra={
        "online_mode": online_mode,
        "trading_enabled": bool(settings.ALPACA_API_KEY),
        "autotrain_enabled": settings.AUTOTRAIN_ENABLED,
        "session_id": session_id,
        "plugins_loaded": loaded_plugins if 'loaded_plugins' in locals() else 0
    })

    # ==== background autotrain setup ====
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    lock_file = base_dir / "autotrain.lock"

    stop_event = threading.Event()

    def start_autotrain() -> tuple[subprocess.Popen, object]:
        if lock_file.exists():
            try:
                pid = int(lock_file.read_text().strip())
                if pid > 0 and Path(f"/proc/{pid}").exists():
                    return None, None
            except Exception:
                pass
            lock_file.unlink(missing_ok=True)

        log_path = log_dir / "autotrain.log"
        log_f = open(log_path, "a")
        proc = subprocess.Popen(
            ["python", "autotrain.py"],
            cwd=str(base_dir),
            stdout=log_f,
            stderr=log_f,
        )
        lock_file.write_text(str(proc.pid))
        return proc, log_f

    def monitor_autotrain(event: threading.Event):
        proc, log_f = start_autotrain()
        while not event.is_set():
            if proc and proc.poll() is not None:
                if log_f:
                    log_f.write(f"AutoTrain exited with {proc.returncode}, restarting...\n")
                    log_f.flush()
                proc, log_f = start_autotrain()
            time.sleep(5)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if log_f:
            log_f.close()
        lock_file.unlink(missing_ok=True)

    monitor_thread = threading.Thread(
        target=monitor_autotrain, args=(stop_event,), daemon=True
    )
    monitor_thread.start()
    reflect_thread = SelfReflection()
    audit_thread = SelfAudit()
    dashboard_thread = TerminalDashboard(audit=audit_thread)
    reflect_thread.start()
    audit_thread.start()
    dashboard_thread.start()

    try:
        while True:
            prompt = input("🧠 You: ").strip()
            if prompt.lower() == "exit":
                logger.info("User requested shutdown")
                print("👋 JARVIS shutting down.")
                break

            start_time = time.time()
            logger.info("User interaction", extra={"prompt": prompt[:100], "online_mode": online_mode})

            # Handle special commands first
            if prompt.lower().startswith("plugin:"):
                # Plugin command
                parts = prompt[7:].strip().split()
                if len(parts) >= 2:
                    command = parts[0]
                    args = parts[1:]
                    response = plugin_manager.execute_command(command, args) or "Plugin command not found"
                else:
                    response = "Usage: plugin:<command> <args>"
            elif prompt.lower().startswith("session:"):
                # Session management
                parts = prompt[8:].strip().split()
                if parts[0] == "list":
                    sessions = conversation_manager.get_recent_sessions(5)
                    response = "Recent sessions:\n" + "\n".join([f"• {s['session_id']}: {s['first_message'][:50]}..." for s in sessions])
                elif parts[0] == "switch" and len(parts) > 1:
                    if conversation_manager.switch_session(parts[1]):
                        response = f"Switched to session: {parts[1]}"
                    else:
                        response = "Session not found"
                else:
                    response = "Usage: session:list or session:switch <session_id>"
            elif online_mode and prompt.lower().startswith("search:"):
                query = prompt.split("search:", 1)[-1].strip()
                logger.info("Web search requested", extra={"query": query[:100]})
                response = web_search(query)
            elif prompt.lower().startswith("trade"):
                _, *symbols = prompt.split()
                logger.info("Trade requested", extra={"symbols": symbols})
                run_autotrader(symbols or None)
                response = "Trade executed"
            else:
                # Try plugins first
                plugin_response = plugin_manager.process_with_plugins(prompt)
                if plugin_response:
                    response = plugin_response
                else:
                    # Build context-aware prompt with conversation history and RAG
                    context_prompt = conversation_manager.build_context_prompt(prompt)
                    rag_context = knowledge_base.get_context_for_query(prompt, max_context_length=500)
                    
                    if rag_context:
                        enhanced_prompt = f"{context_prompt}\n\nRelevant knowledge:\n{rag_context}"
                    else:
                        enhanced_prompt = context_prompt
                    
                    response = brain.ask(enhanced_prompt)

            # Calculate response time
            response_time = time.time() - start_time
            
            # Log to conversation history
            message_type = "command" if prompt.startswith(("plugin:", "session:", "search:", "trade")) else "chat"
            conversation_manager.add_interaction(
                user_message=prompt,
                assistant_response=response,
                message_type=message_type,
                response_time=response_time
            )
            
            if response.startswith("[Error"):
                dashboard_thread.fail += 1
                logger.warning("Error response generated", extra={"response": response[:200]})
            else:
                dashboard_thread.success += 1
                logger.info("Successful response generated", extra={
                    "response_length": len(response),
                    "response_time": response_time,
                    "message_type": message_type
                })
            
            dashboard_thread.log_interaction(prompt, response)
            print(f"🤖 JARVIS: {response}")
            print(f"⏱️ Response time: {response_time:.2f}s")
    except KeyboardInterrupt:
        print("\n👋 JARVIS shutting down.")
    finally:
        stop_event.set()
        reflect_thread.stop()
        audit_thread.stop()
        dashboard_thread.stop()
        monitor_thread.join()
        reflect_thread.join()
        audit_thread.join()
        dashboard_thread.join()
