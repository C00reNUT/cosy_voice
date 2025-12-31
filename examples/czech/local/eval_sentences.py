#!/usr/bin/env python3
"""Czech evaluation sentences for TTS generation during training.

Contains 15 sentences from Sherlock Holmes Czech translation.
Micromamba env: cosyvoice
"""

CZECH_EVAL_SENTENCES = [
    # Sentence 1: Title with Czech diacritics
    "SHERLOCK HOLMES: STUDIE V ŠARLATOVÉ - Kapitola první: PAN SHERLOCK HOLMES.",

    # Sentence 2: Good for testing ř, č sounds
    "Když jsem tam studia dokončil, byl jsem řádně přidělen k Pátému pluku northumberlandských střelců jako asistent chirurga.",

    # Sentence 3: Tests various Czech vowels
    "Pluk v té době pobýval v Indii, a než jsem se k němu mohl připojit, vypukla druhá afghánská válka.",

    # Sentence 4: Tests long sentences
    "Po vylodění v Bombají jsem se dozvěděl, že můj sbor již prošel průsmyky a nachází se hluboko v nepřátelském území.",

    # Sentence 5: Complex sentence structure
    "Následoval jsem jej však s mnoha dalšími důstojníky, kteří byli v téže situaci jako já, a podařilo se nám bezpečně dorazit do Kandaháru, kde jsem nalezl svůj pluk a ihned se ujal svých nových povinností.",

    # Sentence 6: Tests ě, š sounds
    "Tažení přineslo mnohým pocty a povýšení, leč mně nic než neštěstí a pohromy.",

    # Sentence 7: Tests ž sound
    "Byl jsem odvelen od své brigády a přidělen k berkshirskému pluku, s nímž jsem sloužil v osudné bitvě u Majvandu.",

    # Sentence 8: Tests complex consonant clusters
    "Tam mě zasáhla střela z džezailu do ramene, roztříštila kost a škrábla podklíčkovou tepnu.",

    # Sentence 9: Tests ů sound
    "Byl bych padl do rukou vražedných Gázíů, nebýt oddanosti a odvahy, kterou prokázal můj sluha Murray; přehodil mě přes soumara a podařilo se mu dopravit mě bezpečně k britským liniím.",

    # Sentence 10: Tests long á, í sounds
    "Ztrápen bolestí a zesláblý dlouhotrvajícími útrapami, jež jsem podstoupil, byl jsem s velkým vlakem raněných trpitelů odsunut do základní nemocnice v Péšávaru.",

    # Sentence 11: Tests ý sound and complex structures
    "Zde jsem se zotavil a můj stav se zlepšil natolik, že jsem byl schopen procházet se po nemocničních pokojích a dokonce se trochu vyhřívat na verandě, když mě skolil střevní tyfus, ona kletba našich indických držav.",

    # Sentence 12: Tests complex sentence with múltiple clauses
    "Po měsíce se o mém životě pochybovalo, a když jsem konečně přišel k sobě a počal se uzdravovat, byl jsem tak slabý a vyhublý, že lékařská komise rozhodla, že se nesmí ztratit ani den a musím být odeslán zpět do Anglie.",

    # Sentence 13: Replacement - Tests čř combination
    "Za takových okolností jsem přirozeně tíhl k Londýnu, oné veliké stoce, do níž jsou neodolatelně stahováni všichni povaleči a zahaleči impéria.",

    # Sentence 14: Replacement - Tests ž, š sounds
    "Tam jsem nějaký čas bydlel v soukromém hotelu na Strandu, veda nepohodlný, nesmyslný život a utráčeje peníze značně volněji, než jsem měl.",

    # Sentence 15: Replacement - Tests various diacritics
    "Stav mých financí se stal tak povážlivým, že jsem si brzy uvědomil, že musím buď opustit metropoli a usadit se kdesi na venkově, nebo provést naprostou změnu ve svém životním stylu.",
]


def get_eval_sentences() -> list:
    """Get list of evaluation sentences.

    Returns:
        List of 15 Czech sentences for TTS evaluation
    """
    return CZECH_EVAL_SENTENCES.copy()


if __name__ == "__main__":
    sentences = get_eval_sentences()
    print(f"Total evaluation sentences: {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"{i:2d}. {s[:60]}...")
