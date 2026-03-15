# ─── Imports ──────────────────────────────────────────────────────────────────
import requests
import threading
import logging
import re
import customtkinter as ctk

import RebuddyPipeline as rbp
from sentence_transformers import SentenceTransformer


# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("SearchApp")


# ─── Classes ──────────────────────────────────────────────────────────────────

class SearchParameterization:
    def __init__(self, title, doi_prefixes, keywords, max_papers, min_year=None):
        self.title        = title
        self.doi_prefixes = [d.strip() for d in doi_prefixes if d and d.strip()]
        self.keywords     = keywords
        self.max_papers   = min(int(max_papers), 100)
        self.min_year     = int(min_year) if min_year else None


class SearchEngine:
    def __init__(self):
        self.model    = None  # Loaded off the main thread via RebuddyPipeline
        self.base_url = "https://api.crossref.org/works"
        self.contact_email = ""
        self.headers  = {"User-Agent": "SearchApp/1.0"}

    def set_contact_email(self, email: str):
        self.contact_email = email.strip()
        if self.contact_email:
            self.headers = {"User-Agent": f"SearchApp/1.0 (mailto:{self.contact_email})"}
        else:
            self.headers = {"User-Agent": "SearchApp/1.0"}

    def load_model(self):
        log.debug("Starting SentenceTransformer model load...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        log.debug("SentenceTransformer model loaded successfully.")

    def get_new_cursor(self, query):
        log.debug(f"Fetching fresh cursor for query: '{query}'")
        params = {"query": query, "cursor": "*", "rows": 20}
        if self.contact_email:
            params["mailto"] = self.contact_email
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            cursor = response.json()["message"]["next-cursor"]
            log.debug(f"New cursor obtained: {cursor[:40]}...")
            return cursor
        except Exception as e:
            log.error(f"Failed to fetch new cursor: {e}")
            return None

    def fetch_crossref(self, query_title, cursor, min_year=None):
        log.debug(f"Fetching Crossref batch | cursor={cursor[:40] if cursor else 'None'}...")
        try:
            params = {"query": query_title, "cursor": cursor, "rows": 10}
            if self.contact_email:
                params["mailto"] = self.contact_email
            if min_year:
                params["filter"] = f"from-pub-date:{min_year}-01-01"
                log.debug(f"Applying year filter: from-pub-date:{min_year}-01-01")
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            data        = response.json()
            items       = data.get("message", {}).get("items", [])
            next_cursor = data.get("message", {}).get("next-cursor") or ""
            log.debug(f"Crossref returned {len(items)} items | next_cursor={'YES' if next_cursor else 'NONE'}")
            return items, next_cursor
        except Exception as e:
            log.error(f"Crossref fetch failed: {e}")
            return [], ""


class ParameterGUI:
    def __init__(self, root):
        self.root   = root
        self.engine = None
        self.user_email = ""
        self.root.geometry("500x680")
        self.root.title("Search Parameterization Input")

        # ── Progress / Status ──
        self.progress_frame = ctk.CTkFrame(root, fg_color="transparent")
        self.progress_frame.pack(fill="x", padx=20, pady=5)

        self.status_label   = ctk.CTkLabel(self.progress_frame, text="Loading model, please wait...", text_color="gray")
        self.status_label.pack(pady=2)

        self.batch_label    = ctk.CTkLabel(self.progress_frame, text="")   # packed on first search
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="")   # packed on first search

        self.progress_bar   = ctk.CTkProgressBar(self.progress_frame, width=380, mode="indeterminate")
        self.progress_bar.pack(pady=4)
        self.progress_bar.start()

        self.email_btn = ctk.CTkButton(root, text="Add Your Email", command=self.prompt_for_email, width=300)
        self.email_btn.pack(pady=(8, 4))

        self.email_label = ctk.CTkLabel(root, text="Email: not set", text_color="orange")
        self.email_label.pack(pady=(0, 10))

        # ── Input Fields ──
        self.title_entry      = ctk.CTkEntry(root, placeholder_text="Enter Title", width=300)
        self.title_entry.pack(pady=12)

        self.doi_entry        = ctk.CTkEntry(root, placeholder_text="Enter DOI Prefixes separated by ;", width=300)
        self.doi_entry.pack(pady=12)

        self.keywords_entry   = ctk.CTkEntry(root, placeholder_text="Enter Keywords", width=300)
        self.keywords_entry.pack(pady=12)

        self.max_papers_entry = ctk.CTkEntry(root, placeholder_text="Enter Max Amount of Papers", width=300)
        self.max_papers_entry.pack(pady=12)

        self.min_year_entry   = ctk.CTkEntry(root, placeholder_text="Min Publication Year (optional, e.g. 2018)", width=300)
        self.min_year_entry.pack(pady=12)

        self.submit_btn       = ctk.CTkButton(root, text="Submit Parameters", command=self.process_input, state="disabled")
        self.submit_btn.pack(pady=16)

        log.debug("Launching model pre-load thread...")
        threading.Thread(target=self._preload_model, daemon=True).start()

    def _preload_model(self):
        try:
            self.engine = rbp.preload_engine()
            self.root.after(0, self._on_model_ready)
        except Exception as e:
            log.exception(f"Model failed to load: {e}")
            self.root.after(0, self._on_model_error, str(e))

    def _on_model_ready(self):
        log.debug("Model ready — enabling Submit button.")
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        if self.user_email:
            self.engine.set_contact_email(self.user_email)
        self.status_label.configure(text="Model ready. Enter parameters and search.", text_color="green")
        self.submit_btn.configure(state="normal")

    def _on_model_error(self, msg):
        self.progress_bar.stop()
        self.status_label.configure(text=f"Model failed to load: {msg}", text_color="red")

    def _is_valid_email(self, email: str) -> bool:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

    def _parse_min_year(self, raw: str):
        """
        Returns (year_int_or_None, error_msg_or_None).
        Accepts empty string (no filter) or a 4-digit year between 1900 and current year.
        """
        import datetime
        value = raw.strip()
        if not value:
            return None, None
        if not re.fullmatch(r"\d{4}", value):
            return None, "Min year must be a 4-digit number (e.g. 2018)."
        year = int(value)
        current_year = datetime.datetime.now().year
        if year < 1900 or year > current_year:
            return None, f"Min year must be between 1900 and {current_year}."
        return year, None

    def prompt_for_email(self):
        dialog = ctk.CTkInputDialog(
            text="Enter your email for Crossref and Unpaywall API requests:",
            title="Contact Email"
        )
        raw_email = dialog.get_input()
        if raw_email is None:
            return

        email = raw_email.strip().lower()
        if not self._is_valid_email(email):
            self.status_label.configure(text="Please enter a valid email address.", text_color="orange")
            return

        self.user_email = email
        rbp.set_contact_email(email)
        if self.engine:
            self.engine.set_contact_email(email)

        self.email_label.configure(text=f"Email: {email}", text_color="green")
        self.status_label.configure(text="Email saved for Crossref and Unpaywall.", text_color="green")
        log.info("Contact email updated for Crossref + Unpaywall.")

    def process_input(self):
        title = self.title_entry.get().strip()
        if not title:
            self.status_label.configure(text="Please enter a title.", text_color="orange")
            return
        if not self.user_email:
            self.status_label.configure(text="Please click 'Add Your Email' before searching.", text_color="orange")
            return

        # ── Validate min year ──
        min_year, year_error = self._parse_min_year(self.min_year_entry.get())
        if year_error:
            self.status_label.configure(text=year_error, text_color="orange")
            return

        raw_dois     = self.doi_entry.get()
        doi_prefixes = [d.strip() for d in raw_dois.split(";") if d.strip()]
        keywords     = self.keywords_entry.get()

        try:
            max_papers = int(self.max_papers_entry.get())
        except ValueError:
            max_papers = 10

        sp_object = SearchParameterization(title, doi_prefixes, keywords, max_papers, min_year)
        log.info(f"Search submitted | title='{title}' | max={sp_object.max_papers} | min_year={sp_object.min_year}")

        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        self.progress_label.configure(text=f"Papers: 0 / {sp_object.max_papers}")
        self.progress_label.pack(pady=2)
        self.batch_label.configure(text="Batch: —")
        self.batch_label.pack(pady=2)
        self.status_label.configure(text="Searching...", text_color="gray")
        self.submit_btn.configure(state="disabled", text="Searching...")

        threading.Thread(target=self._run_search, args=(sp_object,), daemon=True).start()

    def _run_search(self, sp_object):
        rbp.run_pipeline(
            engine            = self.engine,
            sp                = sp_object,
            progress_callback = self.update_progress,
            batch_callback    = self.update_batch,
            error_callback    = self.show_error
        )
        self.root.after(0, self._on_search_complete, sp_object)

    def _on_search_complete(self, sp_object):
        final_count = len(rbp.CURRENT_PAPERS)
        self.status_label.configure(text=f"Done! Found {final_count} papers.", text_color="green")
        self.progress_bar.set(final_count / sp_object.max_papers if sp_object.max_papers > 0 else 1)
        self.submit_btn.configure(state="normal", text="Submit Parameters")
        log.info(f"Search complete | found={final_count} | max={sp_object.max_papers}")

    def update_progress(self, current, maximum):
        self.root.after(0, self._apply_progress, current, maximum)

    def _apply_progress(self, current, maximum):
        self.progress_label.configure(text=f"Papers: {current} / {maximum}")
        self.progress_bar.set(current / maximum if maximum > 0 else 0)

    def update_batch(self, batch_number):
        self.root.after(0, self._apply_batch, batch_number)

    def _apply_batch(self, batch_number):
        self.batch_label.configure(text=f"Batch: #{batch_number}")

    def show_error(self, message):
        self.root.after(0, self._apply_error, message)

    def _apply_error(self, message):
        self.status_label.configure(text=f"Error: {message}", text_color="red")
        self.submit_btn.configure(state="normal", text="Submit Parameters")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    app = ctk.CTk()
    gui = ParameterGUI(app)
    app.mainloop()
