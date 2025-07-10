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
            writer = csv.DictWriter(file, fieldnames=data[0].keys(),delimiter=';')
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
        self._empty_marker = 'Unbekannt'
        # spacy.cli.download("de_core_news_sm")
        # self.nlp = spacy.load("de_core_news_sm")


    def _extract_price_and_currency(self, price_str: str) -> tuple:
        def is_donated(price_str):
            if distance(price_str.strip(), 'Stiftung') <= 1:
                return True
            if distance(price_str.strip(), 'Geschenk') <= 1:
                return True
            return False

        if not price_str or price_str.strip() == '':
            price = 'Unbekannt'

        if is_donated(price_str):
            return 0, 'Deutsche Mark'

        price = re.sub(r'[^\d]', '', price_str)  # Remove non-digit characters

        if 'DM' in price_str or 'Dm' in price_str:
            return price, 'Deutsche Mark'
        if 'M' in price_str:
            return price, 'Reichsmark'

        return price, 'Deutsche Mark'
            

    def _is_bought(self, row: dict) -> bool:
        erworben = row.get('erworben von', '').strip()
        if erworben.lower() == 'stiftung':
            return False
        if not row['Preis'] or row['Preis'].strip() == '':
            return False
        return True

    def _extract_notes(self, row: dict) -> str | None:
        notes = row.get('Literatur')
        if not self._is_bought(row) and row.get('erworben von') != '':
            notes += f"Angaben aus dem Inventarkartenfeld 'erworben von': {row.get('erworben von')}"
        return notes

    def _extract_standort(self, standort: str) -> str: 
        if not standort or standort.strip() == '':
            return self._empty_marker
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

    def _extract_description(self, row: dict) -> str:
        DEFAULT_DESCRIPTION = 'Dieses Schmuckstück ist aus dem historischen Schmuckinventar der Kunstgewerbeschule Pforzheim.'
        beschreibung = row.get('Beschreibung', DEFAULT_DESCRIPTION)
        if not beschreibung or beschreibung.strip() == '':
            beschreibung = DEFAULT_DESCRIPTION
        return beschreibung
    

    def _update_one_entry(self, row: dict) -> dict:
        """
        Update a single entry in the row based on rules.
        """
        unchanged_keys = []
        def get_or_default(row: dict, key: str) -> str:
            value = row.get(key, '')
            if value is None or value.strip() == '':
                return self._empty_marker
            return value

        updated_row = {}
        for k in unchanged_keys:
            updated_row[k] = get_or_default(row, k)

        updated_row['object_title'] = get_or_default(row, 'Titel')
        updated_row['object_type'] = "Schmuck"
        updated_row['inventory_number'] = get_or_default(row, 'Inv. Nr.')

        updated_row['remarks_short'] = get_or_default(row, 'source_file')
        updated_row['remarks_long'] = get_or_default(row, 'Maße')
        updated_row['literature_title1'] = get_or_default(row, 'Literatur')


        updated_row['abode_regular'] = self._extract_standort(row.get('Standort'))
        updated_row["abode_actual"] = "Schmuckmuseum Pforzheim"

        updated_row['material_separate'] = get_or_default(row, 'Material')
        updated_row['object_description'] = self._extract_description(row)

        insurance_value, insurance_value_currency = self._extract_price_and_currency(row.get('Vers.-Wert', ''))
        updated_row['worth_insurance_value'] = insurance_value
        updated_row['worth_insurance_unit'] = insurance_value_currency

        # updated_row['Notizen'] = self._extract_notes(row) or empty_marker
        updated_row['exhibition_name1'] = get_or_default(row, 'Ausstellung')

        updated_row['image_name1'] = get_or_default(row,'Foto Notes')
        updated_row['image_owner1'] = 'Schmuckmuseum Pforzheim'
        updated_row['image_rights1'] = 'RR-R'
        updated_row['image_visible1'] = 'n'
        updated_row['image_main1'] = 'n'


        updated_row['form_designed_when1'] = get_or_default(row, 'Datierung')
        updated_row['form_designed_who1'] = self._empty_marker
        updated_row['form_designed_where1'] = get_or_default(row, 'Herkunft')

        updated_row['acquisition_type'] = self._empty_marker
        updated_row['acquisition_name'] = 'Erwerb'
        updated_row['acquisition_source_name'] = get_or_default(row, 'erworben von')
        updated_row['acquisition_date'] = get_or_default(row, 'am')
        acquisition_price, acquisition_price_currency = self._extract_price_and_currency(row.get('Preis', ''))
        updated_row['acquisition_price'] = acquisition_price
        updated_row['acquisition_price_currency'] = acquisition_price_currency


        return updated_row