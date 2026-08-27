import os
import json
import re
from openai import OpenAI

class LLMService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        self.last_error = None
        
        self.client = None
        # Check if a valid API key is set
        if self.api_key and self.api_key != "your_api_key_here" and len(self.api_key.strip()) > 0:
            try:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                    default_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "NERVE NOC - Network Intelligence Command Center",
                    }
                )
            except Exception as e:
                print(f"Error initializing OpenRouter client: {e}")
                self.client = None
                
    def route_query(self, user_query: str) -> dict:
        """
        Routes the user query to numerical aggregation or semantic search.
        
        Returns:
            dict: {
                "query_type": "numerical" | "semantic",
                "is_network_related": bool,
                "metric": "fault_count" | "severity" | None,
                "location": str | None, (e.g. "location 123")
                "aggregation": "sum" | "max" | "mean" | "count" | None,
                "target_device_id": int | None,
                "refers_to_active_device": bool
            }
        """
        # If client is not available, run local rule-based fallback routing
        if not self.client:
            return self._local_fallback_route(user_query)
            
        system_prompt = (
            "You are an intent classifier for a Telecom Network Disruption dashboard.\n"
            "Analyze the user query and output a JSON object with the following fields:\n"
            "- 'query_type': Either 'numerical' (for calculations, counting, minimums, maximums, totals) or 'semantic' (for explanations, causes, descriptions, details).\n"
            "- 'is_network_related': boolean. False if the query is completely unrelated to telecom network, faults, locations, resources, devices, or log features (e.g. general knowledge questions, capital of countries, coding help, weather).\n"
            "- 'metric': Either 'fault_count', 'severity', or null.\n"
            "- 'location': The location identifier mentioned in the query (e.g., 'location 123' should be mapped to the standard string 'location 123', extract digits if needed), or null if no specific location is mentioned.\n"
            "- 'aggregation': If numerical, specify 'sum', 'max', 'mean', 'count', or null.\n"
            "- 'target_device_id': If the user explicitly asks about a device ID (e.g., 'device #15086', 'device 15086', 'ID 15086'), extract the numeric ID as an integer. Otherwise null.\n"
            "- 'refers_to_active_device': boolean. True if the query explicitly refers to the currently active or inspected device (e.g., 'this device', 'it', 'why is the device critical', 'what preventive action should be taken for it'). Otherwise false.\n"
            "\n"
            "You must output ONLY valid JSON. Do not include markdown code block syntax (like ```json) or any conversational text. Only output the raw JSON string."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: '{user_query}'"}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            # Clean possible markdown wrapping
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n|```$", "", content, flags=re.MULTILINE).strip()
                
            result = json.loads(content)
            
            # Ensure standard location naming
            if result.get("location"):
                loc_match = re.search(r'\d+', result["location"])
                if loc_match:
                    result["location"] = f"location {loc_match.group(0)}"
                    
            # Ensure refers_to_active_device is boolean
            if "refers_to_active_device" not in result:
                result["refers_to_active_device"] = False
            else:
                result["refers_to_active_device"] = bool(result["refers_to_active_device"])
                
            # Ensure target_device_id is int or None
            if result.get("target_device_id"):
                try:
                    result["target_device_id"] = int(result["target_device_id"])
                except ValueError:
                    result["target_device_id"] = None
            else:
                result["target_device_id"] = None
                
            return result
        except Exception as e:
            self.last_error = str(e)
            print(f"OpenRouter routing failed ({e}), falling back to local routing.")
            return self._local_fallback_route(user_query)

    def _local_fallback_route(self, user_query: str) -> dict:
        """Local regex-based heuristic routing when LLM is unavailable."""
        q_lower = user_query.lower()
        
        # 1. Scope check
        related_keywords = [
            "fault", "severity", "location", "log", "feature", "resource", 
            "network", "disruption", "predict", "status", "health", "volume", 
            "incident", "problem", "failing", "error", "device", "inspect", "repair"
        ]
        is_related = any(kw in q_lower for kw in related_keywords)
        
        # Special check for general knowledge triggers
        unrelated_indicators = ["capital of", "weather in", "france", "paris", "programming", "write code"]
        if any(ui in q_lower for ui in unrelated_indicators):
            is_related = False
            
        # 2. Location extraction
        location = None
        loc_match = re.search(r'location\s*[-_]?\s*(\d+)', q_lower)
        if loc_match:
            location = f"location {loc_match.group(1)}"
            is_related = True
            
        # 3. Device ID extraction
        target_device_id = None
        device_match = re.search(r'(?:device|id|tn|#)\s*#?\s*(\d{4,6})', q_lower)
        if device_match:
            target_device_id = int(device_match.group(1))
            is_related = True

        # 4. Refers to active device check
        refers_to_active_device = False
        active_device_keywords = ["this device", "current device", "the device", "why is it", "action for it", "preventive action for current", "inspected device", "this fault", "this incident"]
        if any(adk in q_lower for adk in active_device_keywords):
            refers_to_active_device = True
            is_related = True
            
        # 5. Query Type and Aggregations
        numerical_keywords = ["total", "how many", "sum", "highest", "most", "count", "average", "mean", "max"]
        what_if_keywords = ["what if", "what happens if", "what is the impact if", "impact if", "remains unresolved", "if this fault", "is not fixed"]
        query_type = "semantic"
        aggregation = None
        metric = None
        
        if any(wk in q_lower for wk in what_if_keywords):
            query_type = "what_if"
            is_related = True
        elif any(nk in q_lower for nk in numerical_keywords):
            query_type = "numerical"
            if "total" in q_lower or "sum" in q_lower or "how many" in q_lower:
                aggregation = "sum"
                metric = "fault_count"
            if "highest" in q_lower or "most" in q_lower or "max" in q_lower:
                aggregation = "max"
                metric = "fault_count"
            if "average" in q_lower or "mean" in q_lower:
                aggregation = "mean"
                metric = "fault_count"
                
        return {
            "query_type": query_type,
            "is_network_related": is_related,
            "metric": metric,
            "location": location,
            "aggregation": aggregation,
            "target_device_id": target_device_id,
            "refers_to_active_device": refers_to_active_device
        }

    def generate_answer(self, user_query: str, context_records: list, active_device_context: str = None) -> str:
        """
        Generates final answer using retrieved summaries, active device context, and user query.
        """
        if not self.client:
            return self._local_fallback_answer(user_query, context_records, active_device_context)
            
        system_prompt = (
            "You are an expert telecom network operations engineer assistant for a NOC Command Center.\n"
            "Your role is to explain network device telemetry, predict fault risks, and recommend troubleshooting actions.\n"
            "Answer the user's questions clearly, concisely, and professionally based on the provided active device telemetry and historical records.\n"
            "Always cite specific health scores, predicted severity classes, anomaly impact percentages, and recommended actions when available."
        )
        
        user_content_parts = []
        if active_device_context:
            user_content_parts.append(f"### Active Inspected Device Telemetry:\n{active_device_context}")
            
        if context_records:
            context_str = "\n\n".join([f"Record {i}:\n{item['summary']}" for i, item in enumerate(context_records, 1)])
            user_content_parts.append(f"### Retrieved Network Context Records:\n{context_str}")
            
        user_content_parts.append(f"### Operator Question:\n{user_query}")
        user_content = "\n\n".join(user_content_parts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1
            )
            ans = response.choices[0].message.content.strip()
            
            # Guardrail check: if answer contains safety evaluation output rather than real answer
            if not ans or ans.lower().startswith("user safety:") or ans.lower() == "safe":
                print("Detected guardrail probe response, falling back to local analytical answer.")
                return self._local_fallback_answer(user_query, context_records, active_device_context)
                
            return ans
        except Exception as e:
            self.last_error = str(e)
            print(f"OpenRouter completion failed ({e}), falling back to local response.")
            return self._local_fallback_answer(user_query, context_records, active_device_context)

    def _local_fallback_answer(self, user_query: str, context_records: list, active_device_context: str = None) -> str:
        """Local fallback that formats retrieved records into a structured analytical report."""
        if not context_records and not active_device_context:
            return "I say that there is not enough information in the dashboard data to answer accurately."
            
        lines = []
        if active_device_context:
            lines.append("### 📡 Inspected Device Telemetry & Diagnosis")
            lines.append(active_device_context)
            lines.append("")
            
        if context_records:
            lines.append("### 🔍 Historical Telemetry & Similar Incident Records")
            for i, rec in enumerate(context_records[:2], 1):
                summary_text = rec.get('summary', str(rec))
                lines.append(f"**Record {i}:** {summary_text}")
            lines.append("")
            
        return "\n".join(lines)
