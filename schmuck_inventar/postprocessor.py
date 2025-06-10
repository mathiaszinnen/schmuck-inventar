from Levenshtein import distance
import re
import csv
# import spacy

class PostProcessor:
    def __init__(self, input_csv, output_csv):
        with open(input_csv, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            self.input_data = [row for row in reader]
        self.output_csv = output_csv

    def _write_to_csv(self, data):
        """
        Write the processed data to a CSV file.
        """
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            print(f"Processed data written to {self.output_csv}")
        
    def _remove_title_parts(self, row: dict) -> dict:
        def _remove_one_header(v: str, k: str) -> str:
            match = re.search(fr'{k}\s*:', v, re.IGNORECASE)
            if match:
                v = v[match.end():].strip()
                v = v.split(k)[-1].strip()
            return v

        updated_inventory_data = {}
        for k,v in row.items():
            updated_inventory_data[k] = _remove_one_header(v,k)
        return updated_inventory_data

    def _update_one_entry(self, row: dict) -> dict:
        """
        Update a single entry in the row based on specific rules.
        This method should be overridden by subclasses to implement custom logic.
        """
        return row

    def postprocess(self):
        """
        Post-process the data after OCR.
        """
        updated_data = []
        for row in self.input_data:
            updated_row = self._remove_title_parts(row)
            updated_row = self._update_one_entry(updated_row)
            updated_data.append(updated_row)
        
        self._write_to_csv(updated_data) 

class SchmuckPostProcessor(PostProcessor):
    def __init__(self, input_csv, output_csv):
        super().__init__(input_csv, output_csv)
        # spacy.cli.download("de_core_news_sm")
        # self.nlp = spacy.load("de_core_news_sm")

    def _extract_price_and_currency(self, price_str: str) -> tuple:
        if not price_str or price_str.strip() == '':
            price = 'Unbekannt'
        if distance(price_str.strip(), 'Stiftung') <= 1:
            price = 0
            currency = 'Mark'
        if 'M' in price_str or 'DM' in price_str:
            price = re.sub(r'[^\d]', '', price_str)  # Remove non-digit characters
            currency = 'Mark'
        else:
            price = re.sub(r'[^\d]', '', price_str)  # Default case: remove non-digit characters
            currency = ''
        return price, currency

    def _is_bought(self, row: dict) -> bool:
        erworben = row.get('erworben von', '').strip()
        if erworben.lower() == 'stiftung':
            return False
        if not row['Preis'] or row['Preis'].strip() == '':
            return False
        return True

    def _extract_notes(self, row: dict) -> str | None:
        notes = row.get('Literatur')
        if self._is_bought(row) and row.get('erworben von') != '':
            notes += f". Angaben aus dem Inventarkartenfeld 'erworben von': {row.get('erworben von')}"
        return notes

    def _extract_standort(self, standort: str) -> str: 
        if not standort or standort.strip() == '':
            return 'Unbekannt' 
        return "alter Standort: " + standort 
    

    def _extract_erwerb(self, row: dict) -> list:
        # erworben_doc = self.nlp(row.get('erworben von', ''))
        # persons = [ent.text for ent in erworben_doc.ents if ent.label_ == 'PER']
        # places = [ent.text for ent in erworben_doc.ents if ent.label_ == 'LOC']
        # TODO
        erworben_str = row.get('erworben von')
        preis_str = row.get('Preis')
        matches = re.search("Hersteller|Entwurf|Ausführung|Herst.|Entw.|Ausf.", erworben_str, flags=re.IGNORECASE)
        if not matches:
            row = row
        return row

        

    def _update_one_entry(self, row: dict, empty_marker='') -> dict:
        """
        Update a single entry in the row based on rules.
        """
        unchanged_keys = ['source_file', 'Ausstellungen']

        updated_row = {}
        for k in unchanged_keys:
            updated_row[k] = row.get(k, empty_marker) 


        updated_row['Objektname'] = row.get('Gegenstand', empty_marker)
        updated_row['Objektart'] = "Schmuck"
        updated_row['Inventarnummer'] = row.get('Inv. Nr.', empty_marker)
        updated_row['Eigentlicher Standort'] = self._extract_standort(row.get('Standort', empty_marker))
        updated_row["aktueller Standort"] = "Schmuckmuseum Pforzheim"
        price, currency = self._extract_price_and_currency(row.get('Preis', ''))
        updated_row['Preis des Erwerbs'] = price
        updated_row['Währung'] = currency
        updated_row['Material/Technik'] = row.get('Material',empty_marker)
        updated_row['Maße'] = row.get('Maße', empty_marker)
        DEFAULT_DESCRIPTION = 'Dieses Objekt befindet sich im Schmuckmuseum Pforzheim. Beschreibung automatisch generiert.'
        updated_row['Beschreibung'] = row.get('Beschreibung', DEFAULT_DESCRIPTION)
        updated_row['Gekauft, wann?'] = row.get('am', empty_marker) 
        updated_row['Versicherungswert'] = row.get('Vers.-Wert', empty_marker)
        updated_row['source_file'] = row.get('source_file', 'Unbekannt')
        updated_row['Notizen'] = self._extract_notes(row) or empty_marker
        updated_row['Negativ-Nr.'] = row.get('Foto Notes', empty_marker)



        return updated_row