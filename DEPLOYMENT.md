# وثيقة تسليم النشر — RAGLab على خادم المنصة

_الإصدار المسلَّم: الوسم `deploy-handoff-20261002` — كل الأرقام والمسارات أدناه تشير إليه._
_هذه الوثيقة self-contained: لا تحتاجون شيئًا آخر غيرها لتشغيل الخدمة. المرجع التفصيلي الكامل عند الحاجة: `raglab/SERVICE.md`._

---

## 1) ما الذي ستشغلونه

مساعد أسئلة-أجوبة موثّق (RAG) فوق مدونة الصيرفة والقانون البنكي التونسي: خدمة REST واحدة (FastAPI/uvicorn) تسترجع من فهرس متجهي (ChromaDB) وتجيب باقتباسات موثقة من المصادر، مع امتناع صادق عن الأسئلة خارج المدونة. لا قاعدة بيانات خارجية، لا GPU، لا خدمات جانبية — **حاوية واحدة**.

- الصورة: `python:3.11-slim` + تبعيات بايثون (انظر `raglab/requirements-service.txt`).
- المنفذ: `8000` (قابل للتغيير في compose).
- المفاتيح السحابية تُعطى عبر بيئة التشغيل فقط — **لا تُخبز في الصورة أبدًا**.

## 2) المتطلبات

- Docker + docker compose (أي إصدار حديث).
- منفذ واحد حر (8000 افتراضيًا) — الخدمة خلف بوابة المنصة/TLS.
- مفاتيح API عند الحاجة (§4) — الخدمة تقلع حتى بلا مفاتيح وتقول بصراحة أي مفتاح ناقص عبر `GET /keys`.

## 3) خطوات النشر (حرفيًا)

```bash
# 1) الكود — منح الفريق صلاحية قراءة على المستودع، أو بديل بلا وصول للكود:
#    بناء الصورة مرة واحدة ثم دفعها إلى رجستري المنصة (ghcr/harbor) وسحبها هناك.
git clone https://github.com/seif-bkh/RAGLab.git
cd RAGLab
git checkout deploy-handoff-20261002

# 2) البيئة — انسخوا القالب واملؤوه (انظر §4)
cp raglab/.env.example raglab/.env
vi raglab/.env

# 3) التشغيل
docker compose up --build -d
docker compose logs -f raglab        # حتى تروا: index ready / Application startup complete

# 4) التحقق (من داخل الخادم — /health محمية بالتوكن)
curl -H "X-Service-Token: $RAGLAB_SERVICE_TOKEN" http://localhost:8000/health
# الوثائق التفاعلية (OpenAPI): http://localhost:8000/docs
```

**أول فهرسة (مرة واحدة):** `POST /ingest` يبني الفهرس فوق المدونة (339 مقطعًا). تستغرق دقائق وتستهلك نداءات تضمين NVIDIA (~350) أول مرة فقط؛ بعدها الفهرس والذاكرات في أحجام دائمة تنجو من إعادة النشر. تتبعوا التقدم عبر `GET /ingest/status`. (بلا مفتاح NVIDIA: بدّلوا مزوّد التضمين — انظر §4.)

## 4) متغيرات البيئة (ملف `raglab/.env`)

**إلزامي فورًا:**
| المتغير | ملاحظة |
|---|---|
| `RAGLAB_SERVICE_TOKEN` | سرّ مشترك للبوابة — **غيّروا القيمة الافتراضية `change-me…` قبل أي تشغيل**. كل طلب (عدا preflight) يحمل `X-Service-Token`؛ مقارنة زمن-ثابت؛ بدونه 401. |

**المفاتيح (حسب المزوّد المستخدم):**
| المتغير | يفعّل |
|---|---|
| `NVIDIA_API_KEY` | التضمين الافتراضي `nvidia/nemotron-3-embed-1b` |
| `XKIRO_API_KEY` | نموذج الجواب الافتراضي `qwen/qwen3.8-max:free` |
| `GEMINI_API_KEY` أو `GOOGLE_API_KEY` | مسار Google الاحتياطي + مسبار النماذج |

**الأهم مما تبقى (الافتراضيات جيدة — الجدول الكامل في `raglab/SERVICE.md`):**
| المتغير | المعنى | الافتراضي |
|---|---|---|
| `RAGLAB_CORS_ORIGINS` | **قيّدوها على أصل الواجهة عندكم** (مثال: `https://front.example.tn`) | `*` |
| `SUFFICIENCY_FIELDS_ENABLED` | حالة الكفاية على كل جواب (مفعّلة بقرار المالك 2026-10-02) | `1` |
| `ANSWER_SUFFICIENCY_COMMITMENT` | الامتناع قبل النموذج عند غياب الدليل + الإحالة (مفعّلة) | `1` |
| `RAGLAB_ALLOW_PROFILE_SWITCH` | `POST /profile` لتبديل النماذج وقت التشغيل (تحت التوكن) | `1` في compose |
| `RAGLAB_DATA_DIRS` | مجلد المدونة (داخل الصورة `/app/docs`) | جاهز |

## 5) واجهة الـ front عندكم — أهم ما يجب معرفته

- **العقد الكامل**: `raglab/CONTRACT.md` + OpenAPI حي على `/docs` + وصف ذاتي على `GET /config` (النماذج، البعد، كيفية التغيير عبر HTTP).
- **`POST /answer` هو المحطة الرئيسة**، ومنذ تفعيل 2026-10-02 احملوا في الحسبان:
  - `status: "answered"` → معه `evidence_status` (كافٍ/غير كافٍ…)، `requirements_covered/missing`، و`claims` باقتباساتها الموثقة.
  - `status: "refused"` مع `reason: "evidence_insufficient"` → **هذا سلوك صحيح مقصود** (خارج المدونة): اعرضوا `refusal_reason` و`referral`. لا إعادة محاولة تلقائية.
  - `status: "greeting"` → تحيات محلية بلا نموذج.
- **المصادقة من الواجهة**: تروسة `X-Service-Token` على كل طلب (تُترك على preflight فقط — الخدمة تعفي OPTIONS). من `local_front`: ‏`--token <القيمة>` أو `export RAGLAB_SERVICE_TOKEN=<القيمة>`.
- **الاختبار الشامل جاهز**: `python local_front.py --base-url http://<host>:8000 --smoke` — طقم دخان كامل عبر HTTP (كل النقاط + اتساق الحقول الجديدة).
- داخل الحاوية أيضًا مسبار النماذج المجانية عند المزودين الثلاثة: `docker compose exec raglab python models_probe.py`.

## 6) العمليات اليومية

| العملية | كيف |
|---|---|
| سجلات | `docker compose logs -f raglab` |
| حالة الخدمة | healthcheck مدمج في compose (كل 30ث، واعٍ بالتوكن) + `GET /health` + مسار التدقيق `GET /audit` |
| وثيقة جديدة بلا إعادة بناء | `POST /documents?index=true` — تبقى في حجم `raglab-documents` عبر إعادة النشر |
| إعادة فهرسة كاملة | `POST /ingest` ثم `GET /ingest/status` |
| إعادة نشر إصدار جديد | `git fetch && git checkout <وسم-أحدث> && docker compose up --build -d` — الفهرس والذاكرات تنجو (أحجام دائمة) |
| تراجع | ارجعوا للوسم السابق وأعيدوا البناء؛ الفهرس متوافق مع نفس وضع التقطيع |

## 7) الأمان — افترضوه مكملًا لبوابة المنصة، لا بديلًا عنها

1. **غيّروا `RAGLAB_SERVICE_TOKEN`** (إلزامي — compose يقلع بقيمة `change-me` والفريق مسؤول عن استبدالها).
2. **اكسروا `RAGLAB_CORS_ORIGINS`** على أصل الواجهة بدل `*` عند أي تعريض خارج localhost.
3. لا مصادقة مستخدمين نهائية مدمجة (سرّ مشترك واحد) — **ضعوا الخدمة خلف بوابة المنصة** (TLS، مصادقة المستخدمين، حدود المعدل).
4. المفاتيح السحابية لا تُطبع في أي خرج (قاعدة مستقرة: أول 8 أحرف مقنّعة فقط) ولا تدخل الصورة (`.env` في `.dockerignore`).
5. حدود الدفع: `RAGLAB_MAX_DOCUMENT_BYTES` افتراضيًا 20MB لكل وثيقة.

## 8) ما الذي يتحقق تلقائيًا قبل وصول أي إصدار إليكم

كل push يمر عبر: الطقم غير المتصل (373 فحصًا) + **بناء صورة الدوكر نفسها واستيراد وحداتها في CI** (`Docker image build + import smoke`). آخر حالة: أخضر على `84e63d8` فأحدث.

---

## 9) أخطاء شائعة

| العرض | السبب | الحل |
|---|---|---|
| `HTTP 401` من الواجهة | الواجهة لم ترسل التوكن، أو أرسلت قيمة مختلفة عن التي في `raglab/.env` بالخادم | `--token <القيمة>` أو `export RAGLAB_SERVICE_TOKEN=...` في شل الواجهة. القيمة المتوقعة = سطر `RAGLAB_SERVICE_TOKEN` في `raglab/.env` على الخادم |
| عدّلتُ `raglab/.env` و401 مستمر | الحاوية العاملة تحتفظ ببيئة الإقلاع — تعديل `.env` لا يحدّثها | `docker compose up -d` لإعادة إنشاء الحاوية، ثم أعد المحاولة |
| (تاريخيًا) التوكن في `.env` لا يُحترم أصلًا | كانت compose تثبّت قيمة افتراضية في `environment:` **تتقدم على `.env`** — أُصلح في هذا الإصدار: `raglab/.env` هو المرجع | حدّثوا إلى هذا الوسم وأعيدوا `docker compose up --build -d` |
| تحذير إقلاع: switching ON بلا توكن | `RAGLAB_ALLOW_PROFILE_SWITCH=1` بلا `RAGLAB_SERVICE_TOKEN` = خدمة مفتوحة | ضعوا التوكن في `raglab/.env` وأعيدوا الإنشاء، أو اضبطوا `RAGLAB_ALLOW_PROFILE_SWITCH=0` |

_لأي سلوك غير متوقع: أرسلوا خرج `local_front.py --smoke` + `docker compose logs --tail 200 raglab` مع الوسم المُشغَّل._
