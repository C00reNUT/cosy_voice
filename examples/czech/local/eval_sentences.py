#!/usr/bin/env python3
"""Czech evaluation sentences for TTS generation during training.

Contains 15 sentences from Sherlock Holmes Czech translation.
Micromamba env: cosyvoice
"""

CZECH_EVAL_SENTENCES = [
    "SHERLOCK HOLMES: STUDIE V ŠARLATOVÉ - Kapitola první: PAN SHERLOCK HOLMES.",

    "Roku osmnáct set sedmdesát osm jsem dosáhl hodnosti doktora medicíny na Londýnské univerzitě a odebral se do Netley, abych absolvoval kurz předepsaný pro vojenské chirurgy.",

    "Když jsem tam studia dokončil, byl jsem řádně přidělen k Pátému pluku northumberlandských střelců jako asistent chirurga.",

    "Pluk v té době pobýval v Indii, a než jsem se k němu mohl připojit, vypukla druhá afghánská válka.",

    "Po vylodění v Bombají jsem se dozvěděl, že můj sbor již prošel průsmyky a nachází se hluboko v nepřátelském území.",

    "Následoval jsem jej však s mnoha dalšími důstojníky, kteří byli v téže situaci jako já, a podařilo se nám bezpečně dorazit do Kandaháru, kde jsem nalezl svůj pluk a ihned se ujal svých nových povinností.",

    "Tažení přineslo mnohým pocty a povýšení, leč mně nic než neštěstí a pohromy.",

    "Byl jsem odvelen od své brigády a přidělen k berkshirskému pluku, s nímž jsem sloužil v osudné bitvě u Majvandu.",

    "Tam mě zasáhla střela z džezailu do ramene, roztříštila kost a škrábla podklíčkovou tepnu.",

    "Byl bych padl do rukou vražedných Gázíů, nebýt oddanosti a odvahy, kterou prokázal můj sluha Murray; přehodil mě přes soumara a podařilo se mu dopravit mě bezpečně k britským liniím.",

    "Ztrápen bolestí a zesláblý dlouhotrvajícími útrapami, jež jsem podstoupil, byl jsem s velkým vlakem raněných trpitelů odsunut do základní nemocnice v Péšávaru.",

    "Zde jsem se zotavil a můj stav se zlepšil natolik, že jsem byl schopen procházet se po nemocničních pokojích a dokonce se trochu vyhřívat na verandě, když mě skolil střevní tyfus, ona kletba našich indických držav.",

    "Po měsíce se o mém životě pochybovalo, a když jsem konečně přišel k sobě a počal se uzdravovat, byl jsem tak slabý a vyhublý, že lékařská komise rozhodla, že se nesmí ztratit ani den a musím být odeslán zpět do Anglie.",

    "Byl jsem tudíž vypraven na vojenské lodi Orontes a o měsíc později jsem vystoupil na přístavní molo v Portsmouthu se zdravím nenávratně zničeným, avšak s povolením otcovské vlády strávit příštích devět měsíců pokusy o jeho nápravu.",

    "Neměl jsem v Anglii příbuzných ani přátel, a byl jsem tedy volný jako pták, či tak volný, jak to jen příjem jedenácti šilinků a šesti pencí denně muži dovolí.",
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
