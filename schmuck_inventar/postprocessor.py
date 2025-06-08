from Levenshtein import distance
import re

class PostProcessor:

    def _remove_headers(self, inventory_data: dict) -> dict:
        def _remove_one_header(v: str, k: str) -> str:
            for key_part in [k,"Gewicht"]:
                match = re.search(fr'{key_part}\s*:', v, re.IGNORECASE)
                if match:
                    v = v[match.end():].strip()
                    v = v.split(key_part)[-1].strip()
            return v

        updated_inventory_data = {}
        for k,v in inventory_data.items():
            updated_inventory_data[k] = _remove_one_header(v,k)
        return updated_inventory_data


    def postprocess(self, inventory_data):
        """
        Post-process the data after OCR.
        This method can be overridden by subclasses to implement custom post-processing logic.
        """
        data = self._remove_headers(inventory_data)
        return data