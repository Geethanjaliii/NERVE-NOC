import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default to the models directory in the project structure
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            
        self.faiss_path = os.path.join(models_dir, "faiss_index.index")
        self.summaries_path = os.path.join(models_dir, "location_summaries.pkl")
        self.faiss_devices_path = os.path.join(models_dir, "faiss_devices.index")
        self.device_summaries_path = os.path.join(models_dir, "device_summaries.pkl")
        self.faiss_playbooks_path = os.path.join(models_dir, "faiss_playbooks.index")
        self.playbook_summaries_path = os.path.join(models_dir, "playbook_summaries.pkl")
        
        if (not os.path.exists(self.faiss_path) or 
            not os.path.exists(self.summaries_path) or 
            not os.path.exists(self.faiss_devices_path) or 
            not os.path.exists(self.device_summaries_path)):
            raise FileNotFoundError(
                f"Vector database files not found. Please run the build script first.\n"
                f"Missing files: {self.faiss_path}, {self.summaries_path}, {self.faiss_devices_path}, {self.device_summaries_path}"
            )
            
        print("Loading SentenceTransformer model locally...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Loading FAISS indices...")
        self.index = faiss.read_index(self.faiss_path)
        self.device_index = faiss.read_index(self.faiss_devices_path)
        
        # Load playbooks if present
        self.playbook_index = None
        self.playbook_summaries = []
        if os.path.exists(self.faiss_playbooks_path) and os.path.exists(self.playbook_summaries_path):
            self.playbook_index = faiss.read_index(self.faiss_playbooks_path)
            with open(self.playbook_summaries_path, "rb") as f:
                self.playbook_summaries = pickle.load(f)
        
        print("Loading summaries mappings...")
        with open(self.summaries_path, "rb") as f:
            self.summaries = pickle.load(f)
            
        with open(self.device_summaries_path, "rb") as f:
            dev_data = pickle.load(f)
            self.device_summaries_dict = dev_data['mapping']
            self.device_summaries_list = dev_data['list']
            
    def retrieve(self, query_text: str, k: int = 5) -> list:
        """
        Encodes query_text and performs L2 nearest-neighbor search for locations in FAISS.
        """
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_records = []
        for idx in indices[0]:
            if 0 <= idx < len(self.summaries):
                retrieved_records.append(self.summaries[idx])
                
        return retrieved_records

    def retrieve_devices(self, query_text: str, k: int = 5) -> list:
        """
        Encodes query_text and performs L2 nearest-neighbor search for devices in FAISS.
        """
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.device_index.search(query_embedding, k)
        
        retrieved_records = []
        for idx in indices[0]:
            if 0 <= idx < len(self.device_summaries_list):
                retrieved_records.append(self.device_summaries_list[idx])
                
        return retrieved_records

    def retrieve_playbooks(self, query_text: str, k: int = 3) -> list:
        """
        Encodes query_text and retrieves matching SOP playbooks from FAISS.
        """
        if not self.playbook_index or not self.playbook_summaries:
            return []
            
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        k_val = min(k, len(self.playbook_summaries))
        distances, indices = self.playbook_index.search(query_embedding, k_val)
        
        retrieved_playbooks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.playbook_summaries):
                retrieved_playbooks.append(self.playbook_summaries[idx])
                
        return retrieved_playbooks

    def cross_reference_fault(self, location: str = None, active_resources: list = None, active_events: list = None, shap_factors: list = None, k_devices: int = 3, k_playbooks: int = 3) -> dict:
        """
        Cross-references an active network incident against both historical device cases and domain SOP playbooks.
        
        Returns:
            dict: {
                'historical_devices': list of matched device summaries,
                'matching_playbooks': list of matched SOP playbooks,
                'query_constructed': str
            }
        """
        # Formulate rich query from active telemetry
        query_parts = []
        if location:
            query_parts.append(f"Location: {location}")
        if active_resources:
            query_parts.append(f"Resources: {', '.join(active_resources)}")
        if active_events:
            query_parts.append(f"Events: {', '.join(active_events)}")
        if shap_factors:
            top_factors = [f[0] for f in shap_factors if f[1] > 0][:3]
            if top_factors:
                query_parts.append(f"Root causes: {', '.join(top_factors)}")
                
        query_str = " | ".join(query_parts) if query_parts else "telecom network fault disruption"
        
        # 1. Retrieve matching historical devices
        hist_devices = self.retrieve_devices(query_str, k=k_devices)
        
        # 2. Retrieve matching SOP playbooks
        playbook_query = query_str
        if active_resources:
            playbook_query += f" {' '.join(active_resources)}"
        if shap_factors:
            playbook_query += f" {' '.join([f[0] for f in shap_factors if f[1] > 0][:2])}"
            
        matching_playbooks = self.retrieve_playbooks(playbook_query, k=k_playbooks)
        
        return {
            'historical_devices': hist_devices,
            'matching_playbooks': matching_playbooks,
            'query_constructed': query_str
        }

    def lookup_device_id(self, device_id: int) -> dict:
        """
        O(1) dictionary lookup for a specific device ID.
        """
        try:
            d_id = int(device_id)
        except (ValueError, TypeError):
            return None
            
        if d_id in self.device_summaries_dict:
            res = self.device_summaries_dict[d_id]
            return {
                'device_id': d_id,
                'summary': res['summary'],
                'location': res['location']
            }
        return None
