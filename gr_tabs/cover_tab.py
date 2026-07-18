import gradio as gr
import logging
from base64 import b64decode
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from libs.fb2_processor import FB2Processor
from libs.utils import now_dir, data_path

logger = logging.getLogger(__name__)

processor = FB2Processor()
def convert_to_jpg(image, dest_image):
    img = Image.fromarray(image)
    img.save(str(dest_image))
    return img

def load_image(ab_name):
    book_path = data_path / ab_name
    destination_path = book_path / 'cover.jpg'
    
    if not destination_path.exists():
        book_path.mkdir(parents=True, exist_ok=True)
        try:
            get_cover_image(ab_name)
        except Exception as e:
            logger.error(f"Не удалось загрузить обложку: {e}")
    
    if destination_path.exists():
        try:
            img = Image.open(str(destination_path))
            return img
        except Exception as e:
            logger.error(f"Не удалось открыть обложку: {e}")
            # Fallback: return empty image placeholder
            img = Image.new('RGB', (200, 300), color='gray')
            return img
    # Fallback: return empty image placeholder
    img = Image.new('RGB', (200, 300), color='gray')
    return img

def add_text_cover(output_path, autor, title):
    image_path = now_dir / 'libs' / 'cover.jpg'
    image = Image.open(str(image_path))
    draw = ImageDraw.Draw(image)
    
    font1_path = now_dir / 'libs' / 'Horovod-Regular.ttf'
    font2_path = now_dir / 'libs' / 'Horovod-Regular.ttf'
    
    font1 = ImageFont.truetype(str(font1_path), size=22)
    font2 = ImageFont.truetype(str(font2_path), size=26)
    
    bbox = draw.textbbox((0, 0), autor, font=font1)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image.width - text_width) / 2
    y = (image.height - text_height) / 3
    draw.text((x, y), autor, font=font1, fill='black')
    
    bbox = draw.textbbox((0, 0), title, font=font2)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image.width - text_width) / 2
    y = (image.height - text_height) / 3 + 24
    draw.text((x, y), title, font=font2, fill='black')

    image.save(str(output_path))

def save_cover_image(ab_name, cv_img):
    book_path = data_path / ab_name
    book_path.mkdir(parents=True, exist_ok=True)
    destination_path = book_path / 'cover.jpg'
    
    if destination_path.exists():
        destination_path.unlink()
    
    gr.Info("Файл загружен", duration=4)
    return convert_to_jpg(cv_img, destination_path)

def get_cover_image(ab_name):
    book_path = data_path / ab_name
    book_path.mkdir(parents=True, exist_ok=True)
    file_path = book_path / f'{ab_name}.fb2'
    
    root = processor.remove_namespaces(str(file_path))
    cover_id = root.xpath("string(//coverpage/image/@href)")
    cover_id = cover_id[1:] if cover_id else 'cover.jpg'
    cover_img = root.xpath(f"string(./binary[@id='{cover_id}'])")
    cover_file = book_path / 'cover.jpg'
    
    if cover_img is not None and cover_img.strip():
        imgdata = b64decode(cover_img)
        cover_file.write_bytes(imgdata)
        return load_image(ab_name)
    else:
        image_path = now_dir / 'libs' / 'cover.jpg'
        if not image_path.exists():
            logger.warning(f"Файл заглушки обложки не найден: {image_path}")
            img = Image.new('RGB', (200, 300), color='gray')
            img.save(str(cover_file))
            return img
        cover_file.write_bytes(image_path.read_bytes())
        return None

def gen_cover(ab_name):
    book_path = data_path / ab_name
    file_path = book_path / f'{ab_name}.fb2'
    
    if not file_path.exists():
        gr.Warning(f"Файл {file_path} не найден")
        return load_image(ab_name)
    
    desc = processor.remove_namespaces(str(file_path))
    first_name = desc.xpath('string(//description/title-info/author/first-name)')
    last_name = desc.xpath('string(//description/title-info/author/last-name)')
    book_title = desc.xpath('string(//description/title-info/book-title)')

    output_path = book_path / 'cover.jpg'
    add_text_cover(
        str(output_path),
        f"{first_name} {last_name}",
        book_title
    )
    return load_image(ab_name)

def cover_tab(ab_path, ab_state):
    with gr.Tab("Обложка", id=0) as cv_tab:
        with gr.Row():
            cur_image = gr.State()
            cover_image = gr.Image(interactive=True, sources=['upload', 'clipboard'])
        with gr.Row():
            cover_from_fb2 = gr.Button("Получить изображение из FB2")
            text_button = gr.Button("Подписать изображение")

    ab_state.change(
        fn=load_image,
        inputs=ab_path,
        outputs=cover_image
    )
    cv_tab.select(
        fn=load_image,
        inputs=ab_path,
        outputs=cover_image
    )
    cover_image.upload(
        fn=save_cover_image,
        inputs=[ab_path, cover_image],
        outputs=cover_image
    )
    cover_from_fb2.click(
        get_cover_image,
        inputs=ab_path,
        outputs=cover_image
    )
    text_button.click(
        gen_cover,
        inputs=ab_path,
        outputs=cover_image
    )