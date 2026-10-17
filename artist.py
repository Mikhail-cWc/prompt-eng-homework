"""Супергерои — проклятие: код, целиком включаемый в Agent_Artist.ipynb."""

# %% Настройки и контракты обмена
import base64
import hashlib
import io
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path.cwd()
OUTPUTS = ROOT / "outputs"
STYLE = (
    "Коллекция «Супергерои — проклятие». Мрачная сатира на культ супергероев "
    "в духе The Boys. Визуальный ориентир — присланный пользователем агитационный "
    "постер: угольно-чёрный, киноварно-красный, грязный кремовый цвет бумаги; "
    "высокий контраст, грубая растровая печать, ксероксный фотомонтаж, потёртости, "
    "сгибы и оборванные края. Вертикальный плакат 2:3. Монументальные фигуры или "
    "символы сверхсилы подавляют масштаб города. Красная геометрия показывает угрозу: "
    "защитник становится источником страха, власти и эксплуатации. "
    "Оригинальные вымышленные герои без эмблем известных франшиз, без портретного "
    "сходства с актёрами. Никаких букв, надписей, логотипов и водяных знаков: "
    "названия будут рядом с изображениями. Без натуралистических ран; угроза "
    "передаётся композицией и последствиями. Не делай героическую рекламу или "
    "глянцевый современный постер: нужен тревожный антипропагандистский плакат."
)
ARTWORKS = [
    {"id": "01-surveillance", "title": "Под защитой", "theme":
     "Аллегорический предметный плакат без людей: над пустым городским макетом "
     "висит огромный защитный щит вымышленного супергероя. Его длинная красная тень "
     "принимает форму решётки, накрывающей улицы. Обещание безопасности превращается "
     "в контроль. Единственный центральный предмет — щит, без лиц, глаз, лазеров, "
     "персонажей, эмблем и текста. Сатира читается в противоречии формы щита и тени."},
    {"id": "02-collateral", "title": "Цена спасения", "theme":
     "Публичный триумф супергероя скрывает ущерб обычным людям. Вверху огромный "
     "улыбающийся герой позирует фотографам; его красный плащ нависает над разрушенным "
     "жилым кварталом и становится трещиной, разделяющей дом. Внизу один человек "
     "держит ключ перед пустым дверным проёмом. Ущерб показан через утрату дома, без тел и крови."},
    {"id": "03-subscription", "title": "Герой по подписке", "theme":
     "Спасение как товар. Огромная рука супергероя протянута к маленькому человеку "
     "над пропастью, но между ладонями установлен турникет с прорезью для монеты. "
     "Красная перекладина преграждает доступ к помощи. За героем — безликие "
     "корпоративные фигуры. Важно ясно показать условность помощи без надписей и ценников."},
    {"id": "04-cult", "title": "Культ на конвейере", "theme":
     "Предметный натюрморт о фабрике популярности: огромный гипсовый бюст "
     "вымышленного улыбающегося супергероя вверху. Под ним конвейер серийно "
     "выпускает одинаковые пустые карнавальные маски с его улыбкой. Один экземпляр "
     "перевёрнут: вместо лица видна безличная внутренняя поверхность. Красная лента "
     "конвейера связывает бюст и маски. В кадре нет живых людей. Смысл — массовое "
     "производство поклонения и отказ от готового образа, ирония сувенирной фабрики."},
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Brief(StrictModel):
    concept: str = Field(min_length=1)
    metaphor: str = Field(min_length=1)
    focal_object: str = Field(min_length=1)
    composition: str = Field(min_length=1)
    avoid: list[str]


class Proposal(StrictModel):
    reply_to_curator: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    short_prompt_en: str = Field(min_length=1)
    changes: list[str]


class Critique(StrictModel):
    observed: str = Field(min_length=1)
    strengths: list[str]
    issues: list[str]
    revision: str = Field(min_length=1)
    alignment: int = Field(ge=0, le=10)
    composition: int = Field(ge=0, le=10)
    originality: int = Field(ge=0, le=10)


class Selection(StrictModel):
    selected: Literal[1, 2]
    reason: str = Field(min_length=1)
    remaining_issues: list[str]


def parse_json(text, schema):
    """JSON проверяется локально; неверные оценки и версии не принимаются."""
    cleaned = text.strip()
    # Qwen иногда включает reasoning-раздел в content вместо отдельного поля.
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    return schema.model_validate(json.loads(cleaned))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_client():
    load_dotenv(ROOT / ".env")
    folder = os.environ.get("folder_id")
    key = os.environ.get("api_key")
    if not folder or not key:
        raise RuntimeError("Заполните folder_id и api_key в локальном .env")
    return OpenAI(base_url="https://ai.api.cloud.yandex.net/v1", api_key=key,
                  project=folder, timeout=240, max_retries=2), folder


# %% Агенты: разные инструкции, модели и собственная память
class Agent:
    """Текст — Responses API, пиксели — мультимодальный Chat Completions API.

    История агента хранится локально и явно передаётся в следующие запросы.
    Это позволяет возобновлять работу без зависимости от срока хранения response_id.
    Визуальный агент получает сами PNG через data URL, а не только описание сцены.
    """

    def __init__(self, name, model, instructions, client, folder, directory, vision=False):
        self.name, self.model, self.instructions = name, model, instructions
        self.client, self.folder, self.directory, self.vision = client, folder, directory, vision
        self.history = []

    def ask(self, step, prompt, schema, images=()):
        path = self.directory / f"{step}.json"
        if path.exists():
            record = load_json(path)
            if (record["input"] != prompt or
                    record["model"].removesuffix("/latest") != self.model.removesuffix("/latest")):
                raise ValueError(f"Изменился запрос в {step}; используйте новый каталог результатов")
            result = schema.model_validate(record["parsed"])
            self.history.extend([{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": record["output"]}])
            return result
        instructions = self.instructions + (
            "\nВерни только JSON, без markdown. Все текстовые поля по-русски, кроме "
            "short_prompt_en. Пиши компактно. Верни экземпляр объекта с заполненными "
            "значениями, не саму JSON Schema. Поля ответа должны находиться в корне: "
            + ", ".join(schema.model_fields)
            + ". Не оборачивай ответ в properties, не возвращай type, title или required. "
            "Схема для проверки ответа:\n" + json.dumps(schema.model_json_schema())
        )
        started = time.monotonic()
        if self.vision:
            # Короткий контракт оказался устойчивее большой JSON Schema у этой VLM.
            contract = (
                '{"selected": 1 или 2, "reason": "обоснование", '
                '"remaining_issues": ["недостаток выбранной картины"]}'
                if schema is Selection else
                '{"observed": "что видно", "strengths": ["достоинство"], '
                '"issues": ["недостаток"], "revision": "конкретная правка", '
                '"alignment": 0-10, "composition": 0-10, "originality": 0-10}'
            )
            instructions = self.instructions + " Верни только JSON вида: " + contract
            content = [{"type": "text", "text": instructions + "\n" + prompt}]
            for image in images:
                encoded = base64.b64encode(image.read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + encoded}})
            response = self.client.chat.completions.create(
                model=f"gpt://{self.folder}/{self.model}",
                # Независимая оценка текущих пикселей без якорения на прошлых баллах.
                messages=[{"role": "user", "content": content}],
                max_tokens=3000, temperature=0.2,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if response.choices[0].finish_reason != "stop":
                raise RuntimeError("VLM не завершила ответ; результат не засчитан")
            output = response.choices[0].message.content or ""
        else:
            response = self.client.responses.create(
                model=f"gpt://{self.folder}/{self.model}", instructions=instructions,
                input=[*self.history, {"role": "user", "content": prompt}],
                max_output_tokens=6000, store=False,
            )
            if response.status != "completed":
                raise RuntimeError("Responses API вернул незавершённый ответ")
            output = response.output_text
        try:
            result = parse_json(output, schema)
        except ValueError:
            save_json(self.directory / f"failed-{step}.json", {
                "agent": self.name, "model": self.model, "input": prompt,
                "output": output, "response_id": response.id,
                "error": "Ответ не прошёл локальную проверку JSON/schema"})
            raise
        record = {
            "agent": self.name, "model": self.model,
            "api": "chat.completions" if self.vision else "responses",
            "instructions": instructions, "input": prompt,
            "generation_settings": ({"temperature": 0.2, "max_tokens": 3000,
                                     "enable_thinking": False} if self.vision else
                                    {"max_output_tokens": 6000, "store": False}),
            "images": [p.name for p in images], "output": output,
            "parsed": result.model_dump(), "response_id": response.id,
            "usage": response.usage.model_dump() if response.usage else None,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        save_json(path, record)
        self.history.extend([{"role": "user", "content": prompt},
                             {"role": "assistant", "content": output}])
        print(f"  {step}: {self.name} завершил ответ", flush=True)
        return result


def generate_image(client, folder, prompt, path):
    """Images API; существующие изображения не генерируются повторно."""
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        record = load_json(metadata_path)
        if record["prompt"] != prompt:
            raise ValueError("Промпт изменился; выберите новый каталог результатов")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Изображение не соответствует сохранённому SHA-256")
        return record
    started = time.monotonic()
    model = "aliceai-image-art-3.0/latest"
    response = client.images.generate(model=f"art://{folder}/{model}", prompt=prompt,
                                      n=1, size="1024x1536")
    if not response.data or not response.data[0].b64_json:
        raise RuntimeError("Images API не вернул изображение в b64_json")
    data = base64.b64decode(response.data[0].b64_json, validate=True)
    image = Image.open(io.BytesIO(data))
    image.load()
    image.save(path, format="PNG")
    record = {"model": model, "api": "images.generate", "prompt": prompt,
              "requested_size": "1024x1536",
              "size": list(image.size), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "elapsed_seconds": round(time.monotonic() - started, 2),
              "timestamp": datetime.now(UTC).isoformat()}
    save_json(metadata_path, record)
    print(f"  {path.name}: изображение сохранено", flush=True)
    return record


# %% Управление диалогом и ограниченный цикл обратной связи
def run_artwork(artwork_id):
    spec = next((a for a in ARTWORKS if a["id"] == artwork_id), None)
    if spec is None:
        raise ValueError(f"Unknown artwork: {artwork_id}")
    directory = OUTPUTS / artwork_id
    directory.mkdir(parents=True, exist_ok=True)
    client, folder = make_client()
    curator = Agent("Куратор", "deepseek-v4.1-flash/latest",
                    "Ты куратор выставки сатирического плаката. Найди ясную зримую метафору "
                    "темы «Супергерои — проклятие». Одна смысловая ловушка на плакат, "
                    "минимум мелких деталей. Образ должен разоблачать героя, а не восхвалять. "
                    + STYLE,
                    client, folder, directory)
    painter = Agent("Художник", "aliceai-llm/latest",
                    "Ты художник и собеседник куратора. Объясни согласие или возражение, "
                    "затем составь конкретный промпт на 100–160 слов. "
                    "Не перечисляй много предметов, удерживай зрительный центр. "
                    "short_prompt_en: компактный самостоятельный промпт до 35 английских слов "
                    "для Stable Diffusion. При доработке ответь на замечания критика в changes. "
                    + STYLE, client, folder, directory)
    critic = Agent("Визуальный критик", "qwen3.6-35b-a3b",
                   "Ты требовательный визуальный критик. Судишь по приложенным пикселям. "
                   "Сначала опиши видимое, затем укажи удачи, недостатки и конкретную правку. "
                   "Не выдумывай отсутствующие детали ради соответствия замыслу. "
                   "Проверь, считываются ли угроза и эксплуатация вместо героического восхищения; "
                   "оцени красно-чёрный печатный плакат и отсутствие случайных надписей. "
                   "Оцени смысл, композицию и оригинальность от 0 до 10; это субъективное мнение. "
                   "Описание до 60 слов; по два достоинства и недостатка; одна конкретная правка. "
                   "При сравнении не предпочитай автоматически второй вариант.",
                   client, folder, directory, vision=True)
    print(spec["title"], flush=True)
    brief = curator.ask("01-brief", f"Тема: {spec['theme']}\nСформулируй замысел.", Brief)
    proposal = painter.ask("02-proposal", "Куратор предложил:\n" +
                           brief.model_dump_json() + "\nОбсуди и предложи картину.", Proposal)
    negotiation = curator.ask("03-negotiation", "Художник ответил:\n" +
                              proposal.model_dump_json() +
                              "\nУточни замысел с учётом его предложения.", Brief)
    first = painter.ask("04-prompt-v1", "Уточнение куратора:\n" +
                        negotiation.model_dump_json() +
                        "\nПодготовь окончательный промпт первой версии.", Proposal)
    image1, image2 = directory / "v1.png", directory / "v2.png"
    generate_image(client, folder, first.prompt, image1)
    critique1 = critic.ask("05-critique-v1", "Замысел:\n" + negotiation.model_dump_json() +
                          "\nОцени первую версию картины.", Critique, [image1])
    revised = painter.ask("06-prompt-v2", "Критик посмотрел первую картину:\n" +
                          critique1.model_dump_json() +
                          "\nСохрани замысел, внеси конкретные правки. Верни полный новый промпт.",
                          Proposal)
    generate_image(client, folder, revised.prompt, image2)
    critique2 = critic.ask("07-critique-v2", "Замысел:\n" + negotiation.model_dump_json() +
                          "\nОцени вторую версию; изображение приложено.", Critique, [image2])
    selection = critic.ask("08-selection", "Сравни обе приложенные картины: сначала версия 1, "
                           "потом версия 2. Выбери одну для выставки, обоснуй по видимым различиям "
                           "и отметь оставшиеся недостатки. Замысел:\n" +
                           negotiation.model_dump_json(), Selection, [image1, image2])
    manifest = {**spec, "brief": negotiation.model_dump(),
                "selected": selection.selected, "selection": selection.model_dump(),
                "scores": [critique1.model_dump(), critique2.model_dump()],
                "final_image": f"v{selection.selected}.png",
                "short_prompt_en": (first if selection.selected == 1 else revised).short_prompt_en,
                "stopping_rule": "Две версии, затем сравнительный выбор VLM; бюджет ограничен."}
    save_json(directory / "manifest.json", manifest)
    return manifest


# %% Открытые веса: дополнительная интерпретация замысла локально
def run_openweight(artwork_id="01-surveillance", seed=42):
    import torch
    from diffusers import AutoPipelineForText2Image

    directory = OUTPUTS / artwork_id
    manifest = load_json(directory / "manifest.json")
    target = directory / "sd-turbo.png"
    if target.exists() and target.with_suffix(".json").exists():
        return load_json(target.with_suffix(".json"))
    model_id = "stabilityai/sd-turbo"
    revision = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    started = time.monotonic()
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, variant="fp16",
        use_safetensors=True,
    ).to(device)
    # Attention slicing снижает пиковую память на Apple Silicon.
    pipe.enable_attention_slicing()
    prompt = manifest["short_prompt_en"]
    token_count = len(pipe.tokenizer(prompt).input_ids)
    if token_count > pipe.tokenizer.model_max_length:
        client, folder = make_client()
        adapter = Agent("Художник", "aliceai-llm/latest",
                        "Сократи промпт для SD-Turbo. В short_prompt_en максимум 35 английских "
                        "слов, простая лексика. Сохрани угрозу супергероя, красно-чёрную палитру "
                        "и стиль потёртого агитационного плаката. "
                        "В changes объясни сокращения, остальные поля заполни кратко.",
                        client, folder, directory)
        adapted = adapter.ask("09-openweight-prompt", "Сократи этот промпт:\n" + prompt, Proposal)
        prompt = adapted.short_prompt_en
        token_count = len(pipe.tokenizer(prompt).input_ids)
        if token_count > pipe.tokenizer.model_max_length:
            raise ValueError(f"Промпт SD-Turbo слишком длинный: {token_count} токенов")
    result = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0,
                  height=512, width=512, generator=torch.Generator("cpu").manual_seed(seed))
    if result.nsfw_content_detected and any(result.nsfw_content_detected):
        raise RuntimeError("Открытая модель отклонила изображение")
    result.images[0].save(target)
    record = {"model": model_id, "revision": revision, "device": device, "seed": seed,
              "steps": 4, "guidance_scale": 0.0, "size": [512, 512], "prompt": prompt,
              "prompt_tokens": token_count, "dtype": str(dtype), "torch": torch.__version__,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "elapsed_seconds_including_load": round(time.monotonic() - started, 2),
              "timestamp": datetime.now(UTC).isoformat()}
    save_json(target.with_suffix(".json"), record)
    print(f"Открытая модель: {target}", flush=True)
    return record
