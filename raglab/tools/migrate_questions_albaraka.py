#!/usr/bin/env python3
"""One-shot corpus question migration (2026-09-24): Banque Atlas -> Al Baraka.

1. questions_50.json  — replace the 20 Atlas cases with Al Baraka equivalents
                        (same ids/categories/languages; expected evidence from
                        the new fiche_produit_banque_albaraka_{ar,fr}.md).
2. questions.json     — repoint the 10 Atlas cases (legacy set).
3. questions_v2.json  — create the Phase-2 set: the target-state symptom
                        categories (colloquial/synonyms/implicit/compound/
                        ambiguous + out-of-scope) over the Al Baraka sheets.

Run from raglab/:  python3 tools/migrate_questions_albaraka.py   (stdlib only)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # raglab/
AR = "fiche_produit_banque_albaraka_ar.md"
FR = "fiche_produit_banque_albaraka_fr.md"

# ---------------------------------------------------------------- q50 replacements
Q50 = {
    "q16": ("Quel est le montant du financement accordé dans le cadre du Compte Épargne Omra Al Baraka ?",
            FR, "fr", "Un financement équivalant à 30% de l’épargne"),
    "q17": ("À quelle condition de durée peut-on souscrire l’épargne Omra chez Al Baraka ?",
            FR, "fr", "Souscription à l’épargne pour une durée de 2 ans"),
    "q18": ("Quels justificatifs de revenus sont demandés pour les salariés dans le dossier du Compte Épargne Omra ?",
            FR, "fr", "Les trois dernières fiches de paie"),
    "q19": ("Quel est le tarif annuel de la Carte VISA Classique ?",
            FR, "fr", "Carte VISA Classique | carte | 40 TND"),
    "q20": ("Qui supporte les pertes en cas de résultat négatif dans la Moudharaba ?",
            FR, "fr", "Al Baraka Bank Tunisia supporte intégralement les pertes"),
    "q21": ("Combien coûte l’abonnement mensuel au pack PLATINUM ?",
            FR, "fr", "22,900 TND/mois"),
    "q22": ("Quelle est la durée de remboursement du financement Omra ?",
            FR, "fr", "Remboursement sur une année"),
    "q23": ("Le leasing automobile d’Al Baraka peut-il financer la totalité du prix du véhicule ?",
            FR, "fr", "100 % du prix du véhicule"),
    "q24": ("Quelle est la durée maximale de remboursement du Al Baraka Leasing Automobile ?",
            FR, "fr", "Durée de remboursement pouvant aller jusqu’à 5 ans"),
    "q25": ("Combien je paie chaque trimestre pour la tenue de mon compte d’épargne ?",
            FR, "fr", "Compte d’épargne | compte | 6 u"),
    "q28": ("Quel pourcentage de la valeur de l’épargne le financement Omra Al Baraka peut-il atteindre ?",
            AR, "ar", "30% من قيمة الادخار"),
    "q29": ("Le leasing automobile d’Al Baraka est-il conforme aux principes de la finance islamique ?",
            AR, "ar", "مطابق لمبادئ الصيرفة الإسلامية"),
    "q30": ("Quelle durée minimale d’épargne est exigée pour le produit Omra Al Baraka ?",
            AR, "ar", "ادخار بسنتين على الأقل"),
    "q31": ("How much does the Al Baraka Tech Card cost per year?",
            FR, "fr", "Al Baraka Tech Card PP | carte | 25 TND"),
    "q32": ("In a Moudharaba, who bears the losses when the result is negative?",
            FR, "fr", "supporte intégralement les pertes"),
    "q33": ("What is the annual fee of the international VISA card?",
            FR, "fr", "Carte VISA Internationale PP | carte | 150 TND"),
    "q34": ("What savings product does Al Baraka offer for the pilgrimage?",
            FR, "fr", "produit d’épargne assorti d’un financement sans marge de profit"),
    "q35": ("How much is a certified cheque for a client of the same branch?",
            FR, "fr", "Chèque certifié client même agence | chèque | 16 u"),
    "q36": ("What is the minimum subscription period for the Omra savings product?",
            FR, "fr", "durée de 2 ans au minimum"),
    "q37": ("Unlike Mourabaha, what is the bank not required to disclose in a Moussawama?",
            FR, "fr", "obligation de déclaration du coût d’achat"),
}

Q50_CATS = {  # preserved from the original set (verified before running)
    "q16": "verbatim", "q17": "paraphrase", "q18": "verbatim", "q19": "verbatim",
    "q20": "verbatim", "q21": "paraphrase", "q22": "verbatim", "q23": "paraphrase",
    "q24": "verbatim", "q25": "paraphrase", "q28": "cross-lingual",
    "q29": "cross-lingual", "q30": "cross-lingual", "q31": "cross-lingual",
    "q32": "cross-lingual", "q33": "cross-lingual", "q34": "cross-lingual",
    "q35": "cross-lingual", "q36": "cross-lingual", "q37": "cross-lingual",
}
Q50_LANGS = {
    "q16": "fr", "q17": "fr", "q18": "fr", "q19": "fr", "q20": "fr", "q21": "fr",
    "q22": "fr", "q23": "fr", "q24": "fr", "q25": "fr", "q28": "fr", "q29": "fr",
    "q30": "fr", "q31": "en", "q32": "en", "q33": "en", "q34": "en", "q35": "en",
    "q36": "en", "q37": "en",
}

# ---------------------------------------------------------- legacy questions.json
# (id, question, language, category, expected_lang or None, expected_substring)
QOLD = [
    ("q01", "Quel est le tarif annuel de la carte VISA Classique chez Al Baraka ?",
     "fr", "verbatim", "fr", "40 TND"),
    ("q02", "Le Al Baraka Leasing Automobile est-il réservé aux professionnels ?",
     "fr", "verbatim", "fr", "usage professionnel"),
    ("q05", "Combien coûte la tenue de compte chaque trimestre pour un compte d’épargne chez Al Baraka ?",
     "fr", "paraphrase", None, "6 u"),
    ("q06", "How much do I pay every three months for my savings account at Al Baraka?",
     "en", "paraphrase", None, "6 u"),
    ("q08", "What is the repayment duration of the Omra financing?",
     "en", "cross-lingual", "fr", "une année"),
    ("q09", "Is the Omra financing granted as a share of the savings?",
     "en", "cross-lingual", "ar", "30% من قيمة الادخار"),
    ("q10", "Quel est le pourcentage maximal du coût total du pèlerinage que le financement Omra ne peut dépasser ?",
     "fr", "cross-lingual", "ar", "50% من الكلفة الجملية للعمرة"),
    ("q12", "Which Al Baraka card is the cheapest one?",
     "en", "cross-lingual", "fr", "18 TND"),
    ("q13", "Quelle formule de financement Al Baraka repose sur une association en capital avec le client ?",
     "fr", "paraphrase", None, "MOUCHARAKA"),
    ("q17", "Which product is called \"Compte Épargne Omra Al Baraka\"?",
     "en", "verbatim", "fr", "Compte Épargne Omra Al Baraka"),
]
QOLD_CATS = {"q01": "verbatim", "q02": "verbatim", "q05": "paraphrase",
             "q06": "paraphrase", "q08": "cross-lingual", "q09": "cross-lingual",
             "q10": "cross-lingual", "q12": "cross-lingual", "q13": "paraphrase",
             "q17": "verbatim"}
QOLD_LANGS = {"q01": "fr", "q02": "fr", "q05": "fr", "q06": "en", "q08": "en",
              "q09": "en", "q10": "fr", "q12": "en", "q13": "fr", "q17": "en"}
QOLD_HAS_LANG = {  # whether the original case carried expected_lang
    "q01": True, "q02": True, "q05": False, "q06": False, "q08": True,
    "q09": True, "q10": True, "q12": True, "q13": False, "q17": True,
}

# ------------------------------------------------------------------- questions_v2
V2 = {
    "colloquial": [
        ("c01", "شحال تسعر كارطة الفيزا الكلاسيك عند بنك البركة؟", "ar", AR, "ar",
         "بطاقة فيزا كلاسيك | 40 دينار", None),
        ("c02", "بقداش ينجم يوصل التمويل في ادخار العمرة عند البركة؟", "ar", AR, "ar",
         "30% من قيمة الادخار", None),
        ("c03", "الليزنغ متاع البركة يغطي كامل فلوس الكارية ولا لا؟", "ar", AR, "ar",
         "في حدود 100% من كلفة السيارة", None),
        ("c04", "قداش من سنة لازم ندخر قبل ماياخذو التمويل متاع العمرة؟", "ar", AR, "ar",
         "ادخار بسنتين على الأقل", None),
        ("c05", "واش البنك يشري السلعة و يبيعها ليا بربح متفق عليه من قبل في المرابحة؟", "ar", AR, "ar",
         "هامش ربح محدّد يُتفق عليه منذ البداية", None),
        ("c06", "شحال يسحب البركة على الحساب الادخاري كل ثلاثة شهور؟", "ar", AR, "ar",
         "حساب الادخار | 6 و.خ", None),
        ("c07", "التأمين على الكارية في الاجارة نخلصه كيفاش؟", "ar", AR, "ar",
         "سداد شهري للتامين خلال كامل مدة عقد الاجارة", None),
    ],
    "synonyms": [
        ("s01", "ما هو رسم الاكتتاب السنوي في البطاقة الذهبية الوطنية للأفراد؟", "ar", AR, "ar",
         "البطاقة الذهبية الوطنية للأفراد | 90 دينار", None),
        ("s02", "ما هي الوثيقة المطلوبة من المتقاعدين لإثبات جرايتهم في ملف تمويل العمرة؟", "ar", AR, "ar",
         "شهادة في الانتفاع بجراية تقعد", None),
        ("s03", "كم هو رسم الشيك المصدق إذا كان الزبون من نفس الوكالة؟", "ar", AR, "ar",
         "شيك مصدق لنفس الوكالة | 16 و.خ", None),
        ("s04", "Quel est le coût annuel de la carte de retrait liée au compte d’épargne ?",
         "fr", FR, "fr", "Carte Epargne Al Baraka | carte | 18 TND", None),
        ("s05", "Quels justificatifs de revenus sont exigés des professions libérales pour la demande de financement Omra ?",
         "fr", FR, "fr", "la déclaration unique des revenus (pour les professions libérales)", None),
        ("s06", "ما أقصى نسبة من كلفة السيارة يمكن تمويلها لاقتناء سيارة مهنية لدى بنك البركة؟", "ar", AR, "ar",
         "في حدود 100% من كلفة السيارة", None),
        ("s07", "La banque doit-elle annoncer sa marge dans une vente Moussawama ?",
         "fr", FR, "fr", "aucune obligation de déclaration du coût d’achat", None),
    ],
    "implicit": [
        ("i01", "أنا موظف أريد أداء العمرة عبر منتج ادخار البركة، ما الوثائق المطلوبة وشروط التمويل؟",
         "ar", AR, "ar", None,
         ["آخر ثلاثة كشوفات راتب (للموظفين)", "30% من قيمة الادخار"]),
        ("i02", "مهني أريد اقتناء سيارة عمل بأقل أثر ضريبي ممكن، ماذا يوفر لي بنك البركة؟",
         "ar", AR, "ar", None,
         ["في حدود 100% من كلفة السيارة", "تقليص القاعدة الخاضعة للضريبة"]),
        ("i03", "Je veux épargner deux ans chez Al Baraka puis obtenir un financement de pèlerinage sans coût supplémentaire",
         "fr", FR, "fr", None,
         ["Souscription à l’épargne pour une durée de 2 ans", "sans marge de profit"]),
        ("i04", "ما مخاطر المضاربة على رب المال إذا كانت النتيجة سلبية؟",
         "ar", AR, "ar", None,
         ["يتحمل بنك البركة تونس الخسارة", "بصفته رب المال"]),
        ("i05", "كيف أستفيد من مزايا ضريبية في إجارة المركبات؟",
         "ar", AR, "ar", None,
         ["استرجاع الضريبة على القيمة المضافة على الإيجار المدفوع", "استخلاص مجزأ للقيمة المضافة"]),
        ("i06", "Quelle formule Al Baraka permet d’acquérir un bien professionnel en le louant d’abord ?",
         "fr", FR, "fr", None,
         ["option d’acquisition du bien loué"]),
        ("i07", "ما شروط الاستفادة من تمويل العمرة للمتقاعد؟",
         "ar", AR, "ar", None,
         ["شهادة في الانتفاع بجراية تقعد", "الاشتراك في الادخار لمدة سنتين كحد أدنى"]),
    ],
    "compound": [
        ("m01", "قارن بين رسم بطاقة فيزا كلاسيك وبطاقة الادخار البركة",
         "ar", AR, "ar", None,
         ["بطاقة فيزا كلاسيك | 40 دينار", "بطاقة الادخار البركة | 18 دينار"]),
        ("m02", "Comparez le coût mensuel du pack CONFORT et du pack PLATINUM",
         "fr", FR, "fr", None,
         ["19,135 TND/mois", "22,900 TND/mois"]),
        ("m03", "ما الفرق بين بيع المرابحة وبيع المساومة من حيث إفصاح البنك عن التكلفة وهامش الربح؟",
         "ar", AR, "ar", None,
         ["هامش ربح محدّد يُتفق عليه منذ البداية", "ليس ملزما بالإعلان عن تكلفة الشراء"]),
        ("m04", "Quelle est la durée de remboursement du leasing automobile et du financement Omra ?",
         "fr", FR, "fr", None,
         ["jusqu’à 5 ans", "sur une année"]),
        ("m05", "من يتحمل الخسارة عند نتيجة سلبية في المضاربة وفي المشاركة؟",
         "ar", AR, "ar", None,
         ["يتحمل بنك البركة تونس الخسارة", "تقاسم الخسائر"]),
        ("m06", "ما رسوم اشتراك Banecti للأفراد وخدمة SMS Flash؟",
         "ar", AR, "ar", None,
         ["من 3,5 إلى 7 دينار شهريًا", "2 و.خ شهريًا"]),
        ("m07", "Quels documents faut-il fournir pour un salarié et pour un retraité dans le dossier Omra ?",
         "fr", FR, "fr", None,
         ["fiches de paie", "Attestation de pension pour les retraités"]),
    ],
    "ambiguous": [
        ("a01", "كم تكلفة البطاقة؟", "ar", AR, "ar",
         "البطاقات الإلكترونية", None,
         "الغموض المؤثر: أي بطاقة من اثنتي عشرة بطاقة؛ الإجابة الجيدة تطلب التحديد أو تعرض جدول البطاقات"),
        ("a02", "Quel est le plafond du financement ?", "fr", FR, "fr",
         "Un financement équivalant à 30% de l’épargne", None,
         "Ambiguïté matérielle : plafond du financement Omra (30% de l’épargne, plafonné à 50% du coût) ou du leasing auto (100% du véhicule) ; produit non précisé"),
        ("a03", "كم رسوم الشيك؟", "ar", AR, "ar",
         "الشيكات", None,
         "الغموض المؤثر: أي خدمة شيكات (شيك مصدق لنفس الوكالة أو لوكالة أخرى، الإنذار المسبق، شهادة عدم الخلاص)"),
        ("a04", "Quel pack choisir ?", "fr", FR, "fr",
         "Pack CONFORT", None,
         "Ambiguïté matérielle : le choix dépend des besoins du client ; la bonne réponse compare les trois packs ou demande des précisions"),
        ("a05", "ما هي أفضل صيغة تمويل لمشروعي الجديد؟", "ar", AR, "ar",
         "المشاركة", None,
         "الغموض المؤثر: المشاركة إذا سيساهم العميل برأس مال، أو المضاربة إذا سيساهم بعمله فقط؛ الإجابة الجيدة تستوضح"),
    ],
    "out-of-scope": [
        ("o01", "ما هو رقم هاتف وكالة بنك البركة بنابل؟", "ar"),
        ("o02", "ما هو سعر صرف الدينار التونسي مقابل الدولار اليوم؟", "ar"),
        ("o03", "Quel est le taux d’intérêt du crédit immobilier ?", "fr"),
        ("o04", "كيف أفتح محفظة بيتكوين عند بنك البركة؟", "ar"),
        ("o05", "Où se trouve l’agence Al Baraka à Sousse ?", "fr"),
    ],
}


def main() -> None:
    # -- questions_50.json ----------------------------------------------------
    p50 = HERE / "questions_50.json"
    data = json.loads(p50.read_text(encoding="utf-8"))
    replaced = 0
    for i, case in enumerate(data["cases"]):
        cid = case["id"]
        if cid in Q50:
            q, doc, lang, sub = Q50[cid]
            assert case["category"] == Q50_CATS[cid], f"{cid}: category drift"
            assert case["language"] == Q50_LANGS[cid], f"{cid}: language drift"
            data["cases"][i] = {
                "id": cid, "question": q, "language": Q50_LANGS[cid],
                "category": Q50_CATS[cid], "expected_document": doc,
                "expected_lang": lang, "expected_substring": sub,
            }
            replaced += 1
    assert replaced == len(Q50), f"expected {len(Q50)} replacements, did {replaced}"
    data["_comment"] = (data["_comment"] +
                        " | Revised 2026-09-24: the 20 Banque Atlas cases now target the "
                        "Al Baraka Bank Tunisie sheets (fiche_produit_banque_albaraka_{ar,fr}.md); "
                        "ids/categories/languages unchanged, numbers must be re-baselined.")
    p50.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"questions_50.json: {replaced} cases migrated to Al Baraka")

    # -- questions.json (legacy) ----------------------------------------------
    pold = HERE / "questions.json"
    data = json.loads(pold.read_text(encoding="utf-8"))
    old_by_id = {c["id"]: c for c in data["cases"]}
    for cid, q, lang, cat, elang, sub in QOLD:
        case = old_by_id[cid]
        assert case["category"] == cat, f"{cid}: category drift"
        assert case["language"] == lang, f"{cid}: language drift"
        case["question"] = q
        case["expected_substring"] = sub
        if QOLD_HAS_LANG[cid]:
            case["expected_lang"] = elang
        elif "expected_lang" in case:
            del case["expected_lang"]
    pold.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"questions.json: {len(QOLD)} legacy cases repointed to Al Baraka")

    # -- questions_v2.json ------------------------------------------------------
    cases = []
    for cat in ("colloquial", "synonyms", "implicit", "compound", "ambiguous"):
        for row in V2[cat]:
            if cat == "ambiguous":
                cid, q, lang, doc, elang, sub, subs, note = row
            else:
                cid, q, lang, doc, elang, sub, subs = row
                note = None
            case = {"id": cid, "question": q, "language": lang, "category": cat,
                    "expected_document": doc, "expected_lang": elang}
            if sub is not None:
                case["expected_substring"] = sub
            if subs is not None:
                case["expected_substrings"] = subs
            if note is not None:
                case["ambiguity_note"] = note
            cases.append(case)
    for cid, q, lang in V2["out-of-scope"]:
        cases.append({"id": cid, "question": q, "language": lang,
                      "category": "out-of-scope"})

    v2 = {
        "_comment": ("Phase-2 target-state question set (RAGLAB_ROADMAP.md): the original "
                     "symptom categories over the Al Baraka Bank Tunisie pilot sheets — "
                     "colloquial (Tunisian dialect vs formal wording), synonyms "
                     "(institutional vocabulary), implicit (requirement not named verbatim, "
                     "expected_substrings = the evidence set), compound (explicit "
                     "multi-requirement), ambiguous (materially underspecified; the "
                     "expected evidence is the region any clarification-ready answer must "
                     "show, ambiguity_note documents what a good answer clarifies) and "
                     "out-of-scope. Baseline measured with harness50 (BM25) locally and "
                     "real-test.yml (vector/rrf) in CI."),
        "cases": cases,
    }
    (HERE / "questions_v2.json").write_text(
        json.dumps(v2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"questions_v2.json: {len(cases)} cases written "
          f"({len(V2['colloquial'])} colloquial, {len(V2['synonyms'])} synonyms, "
          f"{len(V2['implicit'])} implicit, {len(V2['compound'])} compound, "
          f"{len(V2['ambiguous'])} ambiguous, {len(V2['out-of-scope'])} out-of-scope)")


if __name__ == "__main__":
    main()
