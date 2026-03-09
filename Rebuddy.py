import tkinter as tk
from tkinter import messagebox, scrolledtext
import json
import threading
import re
import requests
from sentence_transformers import SentenceTransformer, util
from habanero import Crossref
from langgraph.graph import StateGraph
from typing import TypedDict, Optional, List
from datetime import datetime


class SearchEngine:
    """Search Engine object with configurable parameters"""
    def __init__(self, SearchP: str, Abstract: bool = True, DOI_SOURCE: str = "", Keywords: str = "", MaxPapers: int = 10):
        self.SearchP = SearchP.strip()
        self.Abstract = Abstract
        self.DOI_SOURCE = DOI_SOURCE.strip()  # Regex pattern for DOI (e.g., "10.1145" for ACM)
        self.Keywords = Keywords.strip()
        self.MaxPapers = min(MaxPapers, 50)  # Maximum 50 papers
        self.cr = Crossref()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.query_embedding = self.model.encode(f"{self.SearchP} {self.Keywords}".strip(), convert_to_tensor=True)


class PaperState(TypedDict):
    """State object for LangGraph workflow"""
    search_engine: SearchEngine
    query: str
    papers_found: List[dict]
    papers_filtered: List[dict]
    papers_verified: List[dict]
    final_results: List[dict]
    verified_papers_array: List[dict]  # Global array for storing verified papers
    retry_count: int
    step: str
    max_retries: int


class CrossSearch:
    """4-Step LangGraph-based paper search and validation system"""
    
    def __init__(self):
        self.workflow = StateGraph(PaperState)
        self.setup_workflow()
    
    def setup_workflow(self):
        """Setup the 4-step LangGraph workflow"""
        self.workflow.add_node("step1_search", self.step1_search_papers)
        self.workflow.add_node("step2_similarity", self.step2_similarity_comparison)
        self.workflow.add_node("step3_verify", self.step3_extra_data_verify)
        self.workflow.add_node("cleaning_step", self.cleaning_step)
        self.workflow.add_node("step4_output", self.step4_json_output)
        
        # Define edges with conditional logic
        self.workflow.add_edge("step1_search", "step2_similarity")
        self.workflow.add_conditional_edges(
            "step2_similarity",
            self.should_continue,
            {"continue": "step3_verify", "retry": "step1_search", "discard": "step4_output"}
        )
        self.workflow.add_edge("step3_verify", "cleaning_step")
        self.workflow.add_edge("cleaning_step", "step4_output")
        
        self.workflow.set_entry_point("step1_search")
    
    def step1_search_papers(self, state: PaperState) -> PaperState:
        """STEP 1: Find papers online using CrossRef and Habanero"""
        search_engine = state["search_engine"]
        query = f"{search_engine.SearchP} {search_engine.Keywords}".strip()
        
        try:
            # Search CrossRef using habanero
            results = search_engine.cr.works(query=query, limit=20)
            papers = []
            
            for item in results["message"]["items"]:
                title = item.get("title", ["Unknown"])[0] if item.get("title") else "Unknown"
                abstract = item.get("abstract", "") if search_engine.Abstract else ""
                doi = item.get("DOI", "")
                published_date = item.get("published-online", {}).get("date-parts", [[None]])[0][0] or "Unknown"
                keywords = item.get("keywords", [])
                
                # Filter by DOI SOURCE if specified (using regex)
                if search_engine.DOI_SOURCE:
                    if not re.match(search_engine.DOI_SOURCE, doi):
                        continue
                
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "published_date": published_date,
                    "keywords": keywords,
                    "similarity_score": None,
                    "is_open_access": None
                })
            
            state["papers_found"] = papers
            
        except Exception as e:
            state["papers_found"] = []
        
        return state
    
    def step2_similarity_comparison(self, state: PaperState) -> PaperState:
        """STEP 2: Compare papers using cosine similarity (60% threshold) and add to global array"""
        search_engine = state["search_engine"]
        threshold = 0.60
        filtered_papers = []
        
        for paper in state["papers_found"]:
            # Compare title with query
            title_score = util.cos_sim(
                search_engine.query_embedding,
                search_engine.model.encode(paper["title"], convert_to_tensor=True)
            ).item()
            
            paper["similarity_score"] = round(title_score, 3)
            
            # TODO: Future implementation - Add Decision Tree for more sophisticated filtering
            # Consider abstract, keywords, citation count, publication date, etc.
            
            if title_score >= threshold:
                filtered_papers.append(paper)
                # ADD TO GLOBAL ARRAY
                state["verified_papers_array"].append({
                    "title": paper["title"],
                    "doi": paper["doi"],
                    "similarity_score": paper["similarity_score"],
                    "abstract": paper["abstract"],
                    "published_date": paper["published_date"],
                    "is_open_access": None  # Will be filled in step 3
                })
        
        state["papers_filtered"] = filtered_papers
        
        return state
    
    def step3_extra_data_verify(self, state: PaperState) -> PaperState:
        """STEP 3: Verify extra data - Check abstract/date presence and open access status"""
        search_engine = state["search_engine"]
        verified_papers = []
        
        for paper in state["papers_filtered"]:
            # DISCARD: If no abstract or placeholder text
            abstract_lower = paper["abstract"].lower().strip()
            if not abstract_lower or "no abstract available" in abstract_lower or abstract_lower == "abstract" or abstract_lower == "n/a":
                continue
            
            # DISCARD: If no publication date
            if paper["published_date"] == "Unknown":
                continue
            
            # Compare abstract with keywords
            abstract_match = True
            if search_engine.Keywords and paper["abstract"]:
                keywords_list = search_engine.Keywords.split()
                abstract_lower = paper["abstract"].lower()
                abstract_match = any(kw.lower() in abstract_lower for kw in keywords_list)
            
            # Check if paper is open access using Unpaywall API
            is_open_access = False
            if paper["doi"]:
                try:
                    response = requests.get(f"https://api.unpaywall.org/v2/{paper['doi']}?email=user@example.com", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        is_open_access = data.get("is_oa", False)
                except:
                    is_open_access = None
            
            paper["is_open_access"] = is_open_access
            
            # Keep paper if abstract matches keywords (or no keywords specified)
            if abstract_match:
                verified_papers.append(paper)
        
        state["papers_verified"] = verified_papers
        
        return state
    
    def cleaning_step(self, state: PaperState) -> PaperState:
        """CLEANING STEP: Remove duplicates - Check if papers from step 2 & 3 are already in global array"""
        # Get all DOIs already in the global array
        existing_dois = {paper["doi"] for paper in state["verified_papers_array"]}
        
        papers_to_keep = []
        
        # Check papers_verified (from step 3)
        for paper in state["papers_verified"]:
            if paper["doi"] not in existing_dois:
                papers_to_keep.append(paper)
                # Add new paper to global array
                state["verified_papers_array"].append({
                    "title": paper["title"],
                    "doi": paper["doi"],
                    "similarity_score": paper["similarity_score"],
                    "abstract": paper["abstract"],
                    "published_date": paper["published_date"],
                    "is_open_access": paper.get("is_open_access")
                })
        
        state["papers_verified"] = papers_to_keep
        
        return state
    
    def step4_json_output(self, state: PaperState) -> PaperState:
        """STEP 4: Output as formatted JSON using global verified papers array"""
        output_papers = []
        for paper in state["verified_papers_array"]:  # Use global array
            output_papers.append({
                "title": paper["title"],
                "doi": paper["doi"],
                "similarity_score": f"{paper['similarity_score']*100:.1f}%",
                "abstract": paper["abstract"][:200] + "..." if paper["abstract"] else "No abstract available",
                "is_open_access": paper["is_open_access"],
                "published_date": paper["published_date"]
            })
        
        final_output = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_papers": len(output_papers),
            "requested_papers": state["search_engine"].MaxPapers,
            "papers": output_papers
        }
        
        state["final_results"] = output_papers
        return state
    
    def should_continue(self, state: PaperState) -> str:
        """Conditional logic for workflow branching with paper count checking"""
        verified_count = len(state["verified_papers_array"])
        max_papers = state["search_engine"].MaxPapers
        
        # Check if we have enough papers
        if verified_count >= max_papers:
            return "continue"
        else:
            # Check if we can retry
            if state["retry_count"] < state["max_retries"]:
                state["retry_count"] += 1
                return "retry"
            else:
                return "continue"
    
    def execute(self, SearchP: str, Keywords: str = "", DOI_SOURCE: str = "", MaxPapers: int = 10) -> dict:
        """Execute the complete 4-step workflow"""
        search_engine = SearchEngine(
            SearchP=SearchP,
            Abstract=True,
            DOI_SOURCE=DOI_SOURCE,
            Keywords=Keywords,
            MaxPapers=MaxPapers
        )
        
        initial_state: PaperState = {
            "search_engine": search_engine,
            "query": SearchP,
            "papers_found": [],
            "papers_filtered": [],
            "papers_verified": [],
            "final_results": [],
            "verified_papers_array": [],  # Global array
            "retry_count": 0,
            "step": "step1_search",
            "max_retries": 3  # Hard limit for loop
        }
        
        app = self.workflow.compile()
        final_state = app.invoke(initial_state)
        
        output = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_papers": len(final_state["verified_papers_array"]),
            "requested_papers": MaxPapers,
            "papers": [
                {
                    "title": p["title"],
                    "doi": p["doi"],
                    "similarity_score": f"{p['similarity_score']*100:.1f}%",
                    "abstract": p["abstract"][:200] + "..." if p["abstract"] else "No abstract available",
                }
                for p in final_state["verified_papers_array"]
            ]
        }
        
        return output


class CrossSearchGUI:
    """Modular GUI object for CrossSearch"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CrossSearch - Research Paper Finder")
        self.root.geometry("700x550")
        self.searcher = None
        self.parameters_set = False
        self.setup_ui()
    
    def setup_ui(self):
        """Setup GUI components"""
        # Title
        tk.Label(self.root, text="CrossSearch - Research Paper Finder", 
                font=("Arial", 16, "bold"), fg="#2196F3").pack(pady=15)
        
        # Parameters Frame
        params_frame = tk.LabelFrame(self.root, text="Search Parameters", font=("Arial", 11, "bold"), padx=20, pady=10)
        params_frame.pack(padx=20, pady=10, fill=tk.X)
        
        tk.Label(params_frame, text="Search Term:", font=("Arial", 10)).pack(anchor=tk.W)
        self.search_entry = tk.Entry(params_frame, width=50, font=("Arial", 10))
        self.search_entry.pack(anchor=tk.W, pady=5)
        self.search_entry.insert(0, "machine learning")
        
        tk.Label(params_frame, text="Keywords (optional, space-separated):", font=("Arial", 10)).pack(anchor=tk.W)
        self.keywords_entry = tk.Entry(params_frame, width=50, font=("Arial", 10))
        self.keywords_entry.pack(anchor=tk.W, pady=5)
        self.keywords_entry.insert(0, "neural networks deep learning")
        
        tk.Label(params_frame, text="DOI Source (regex pattern, optional):", font=("Arial", 10)).pack(anchor=tk.W)
        self.doi_entry = tk.Entry(params_frame, width=50, font=("Arial", 10))
        self.doi_entry.pack(anchor=tk.W, pady=5)
        self.doi_entry.insert(0, "10\\.1145|10\\.1109")
        
        tk.Label(params_frame, text="Maximum Papers (1-50):", font=("Arial", 10)).pack(anchor=tk.W)
        max_frame = tk.Frame(params_frame)
        max_frame.pack(anchor=tk.W, pady=5)
        self.max_papers_entry = tk.Entry(max_frame, width=10, font=("Arial", 10))
        self.max_papers_entry.pack(side=tk.LEFT)
        self.max_papers_entry.insert(0, "10")
        tk.Label(max_frame, text="(Default: 10, Max: 50)", font=("Arial", 9), fg="#666666").pack(side=tk.LEFT, padx=10)
        
        # Button Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.set_params_button = tk.Button(button_frame, text="Set Parameters", command=self.set_parameters,
                                          bg="#FF9800", fg="white", font=("Arial", 11, "bold"), width=18)
        self.set_params_button.pack(side=tk.LEFT, padx=5)
        
        self.execute_button = tk.Button(button_frame, text="Execute Search", command=self.execute_search,
                                       bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=18, state=tk.DISABLED)
        self.execute_button.pack(side=tk.LEFT, padx=5)
        
        # Status Label
        self.status_label = tk.Label(self.root, text="Status: Ready", font=("Arial", 9), fg="#666666")
        self.status_label.pack(anchor=tk.W, padx=20)
        
        # Output Frame
        tk.Label(self.root, text="Results:", font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=20, pady=(10, 5))
        self.output_text = scrolledtext.ScrolledText(self.root, width=80, height=15, font=("Courier", 9))
        self.output_text.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)
    
    def set_parameters(self):
        """Set parameters and initialize SearchEngine"""
        search_term = self.search_entry.get().strip()
        keywords = self.keywords_entry.get().strip()
        doi_pattern = self.doi_entry.get().strip()
        
        try:
            max_papers = int(self.max_papers_entry.get().strip())
            if max_papers < 1 or max_papers > 50:
                messagebox.showerror("Error", "Maximum papers must be between 1 and 50")
                return
        except ValueError:
            messagebox.showerror("Error", "Maximum papers must be a valid number")
            return
        
        if not search_term:
            messagebox.showerror("Error", "Please enter a search term")
            return
        
        try:
            self.searcher = CrossSearch()
            self.search_term = search_term
            self.keywords = keywords
            self.doi_pattern = doi_pattern
            self.max_papers = max_papers
            self.parameters_set = True
            
            self.status_label.config(text=f"Status: Parameters set ✓ | Search: '{search_term}' | Max: {max_papers} papers", fg="#4CAF50")
            self.set_params_button.config(state=tk.DISABLED)
            self.execute_button.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "✓ Parameters configured successfully!\n\n")
            self.output_text.insert(tk.END, f"Search Term: {search_term}\n")
            self.output_text.insert(tk.END, f"Keywords: {keywords if keywords else '(none)'}\n")
            self.output_text.insert(tk.END, f"DOI Pattern: {doi_pattern if doi_pattern else '(none)'}\n")
            self.output_text.insert(tk.END, f"Maximum Papers: {max_papers}\n\n")
            self.output_text.insert(tk.END, "Click 'Execute Search' to start the 4-step pipeline.\n")
            
            messagebox.showinfo("Success", "Parameters set successfully!\nNow click 'Execute Search' to start.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set parameters: {str(e)}")
    
    def execute_search(self):
        """Execute the search pipeline"""
        if not self.parameters_set or not self.searcher:
            messagebox.showerror("Error", "Please set parameters first")
            return
        
        self.execute_button.config(state=tk.DISABLED)
        self.set_params_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Executing pipeline...", fg="#FF9800")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "🚀 Starting 4-step pipeline...\n\n")
        self.root.update()
        
        # Run search in a separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._search_thread)
        thread.daemon = True
        thread.start()
    
    def _search_thread(self):
        """Execute search in background thread"""
        try:
            results = self.searcher.execute(
                SearchP=self.search_term,
                Keywords=self.keywords if self.keywords else "",
                DOI_SOURCE=self.doi_pattern if self.doi_pattern else "",
                MaxPapers=self.max_papers
            )
            
            # Update GUI from main thread
            self.root.after(0, self._display_results, results)
        
        except Exception as e:
            self.root.after(0, lambda: self._display_error, str(e))
    
    def _display_results(self, results):
        """Display results in the GUI"""
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, json.dumps(results, indent=2))
        
        # Save to file
        try:
            with open("results.json", "w") as f:
                json.dump(results, f, indent=2)
        except:
            pass
        
        self.status_label.config(text=f"Status: Complete ✓ | Found {results['total_papers']} papers", fg="#4CAF50")
        self.execute_button.config(state=tk.NORMAL)
        self.set_params_button.config(state=tk.NORMAL)
        
        messagebox.showinfo("Success", f"✓ Found {results['total_papers']} papers!\nResults saved to results.json")
    
    def _display_error(self, error_msg):
        """Display error in the GUI"""
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"❌ Error occurred:\n\n{error_msg}\n\n")
        import traceback
        self.output_text.insert(tk.END, traceback.format_exc())
        
        self.status_label.config(text="Status: Error occurred", fg="#F44336")
        self.execute_button.config(state=tk.NORMAL)
        self.set_params_button.config(state=tk.NORMAL)
        
        messagebox.showerror("Error", f"Search failed:\n{error_msg}")


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    gui = CrossSearchGUI(root)
    root.mainloop()

