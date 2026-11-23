from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import gradio as gr
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, env):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env


def processor_class(path, data):
    env = dict(re=re, time=time, etree=etree, dataclass=dataclass,
               Dict=Dict, List=List, Tuple=Tuple, Optional=Optional,
               config=SimpleNamespace(), data_path=data,
               logger=logging.getLogger('analysis-fixture'),
               normalize_russian=lambda text: text)
    return load_definitions(path, {'BookMetadata', 'FB2Processor'}, env)['FB2Processor']


def new_processor(cls):
    # No SQL, network, model loading or user-settings writes in these fixtures.
    obj = cls.__new__(cls)
    obj.parser = SimpleNamespace(preprocess=lambda s: s, garbage=lambda s: s)
    obj.stop_parsing = False
    return obj


def write_book(base, name, paragraphs=150):
    project = base / name
    project.mkdir()
    root = etree.Element('FictionBook')
    description = etree.SubElement(root, 'description')
    info = etree.SubElement(description, 'title-info')
    author = etree.SubElement(info, 'author')
    etree.SubElement(author, 'first-name').text = 'Test'
    etree.SubElement(author, 'last-name').text = 'Author'
    etree.SubElement(info, 'book-title').text = 'Fixture'
    section = etree.SubElement(etree.SubElement(root, 'body'), 'section')
    for i in range(paragraphs):
        etree.SubElement(section, 'p').text = (
            f'Абзац {i}: зам+ок. English text. שלום עולם. ' * (1 + i % 4))
    (project / f'{name}.fb2').write_bytes(etree.tostring(root, encoding='utf-8'))
    return project


class ProcessorProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lecta_analysis_')
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.cls = processor_class(ROOT / 'libs/fb2_processor.py', self.data)

    def test_single_section_reports_before_output_exists(self):
        project = write_book(self.data, 'Book')
        p = new_processor(self.cls)
        intermediate = []
        for percent, message in p.process_book('Book', replace=True, sound_effect=False):
            if 'paragraphs' in message and 0 < percent < 99:
                intermediate.append(percent)
                self.assertFalse((project / 'xml/1.xml').exists())
        self.assertGreaterEqual(len(intermediate), 2)
        self.assertEqual(sorted(intermediate), intermediate)
        self.assertTrue((project / 'xml/1.xml').exists())

    def test_stop_preserves_existing_and_stale_xml(self):
        project = write_book(self.data, 'Book')
        xml = project / 'xml'
        xml.mkdir()
        (xml / '1.xml').write_text('<speak>existing</speak>')
        (xml / '2.xml').write_text('<speak>retained</speak>')
        p = new_processor(self.cls)
        rows = []
        for row in p.process_book('Book', replace=True, sound_effect=False):
            rows.append(row)
            if 'paragraphs 50/' in row[1]:
                p.stop_parse()
        self.assertIn('Interrupted', rows[-1][1])
        self.assertLess(rows[-1][0], 100)
        self.assertEqual((xml / '1.xml').read_text(), '<speak>existing</speak>')
        self.assertTrue((xml / '2.xml').exists())

    def test_xml_matches_before_patch_with_same_preprocessing(self):
        # Golden hashes captured from the pre-repair implementation. Keep the
        # regression portable: it must not depend on a local backup directory.
        baseline = {
            (False, False): 'af53253f4a1aedde496e23f4aa5b24423c6a5a9e08f082bd29026abada460d9b',
            (True, False): 'af53253f4a1aedde496e23f4aa5b24423c6a5a9e08f082bd29026abada460d9b',
            (False, True): 'ccf45f39a87402fac79d70ab119ee4814cd0dc36ef3eedcfd150d968c5760bbd',
        }
        for english, remove_ru in ((False, False), (True, False), (False, True)):
            with self.subTest(english=english, remove_ru=remove_ru):
                name = f'Book_{english}_{remove_ru}'
                project = write_book(self.data, name, paragraphs=80)
                kwargs = dict(replace=True, sound_effect=False, ch_size=70,
                              is_english=english, remove_ru=remove_ru)
                list(new_processor(self.cls).process_book(name, **kwargs))
                actual = hashlib.sha256((project / 'xml/1.xml').read_bytes()).hexdigest()
                self.assertEqual(actual, baseline[english, remove_ru])
                self.assertEqual(len(list((project / 'xml').glob('*.xml'))), 1)

    def test_parse_lines_stays_synchronous(self):
        parent = etree.fromstring('<section><p>Один. Two. שלום.</p></section>')
        result = new_processor(self.cls).parse_lines(parent, max_length=8)
        self.assertIsNone(result)
        self.assertGreater(len(parent), 1)

    def test_wrapper_reports_before_constructor_and_does_not_complete_error(self):
        env = dict(gr=gr, Path=Path, data_path=self.data,
                   AppConfig=SimpleNamespace(save_user_settings=mock.Mock()),
                   get_xml_files=lambda _: [],
                   get_parse_metrics_html=lambda p, s: (p, s),
                   FB2Processor=mock.Mock(), processor=None,
                   logger=logging.getLogger('analysis-fixture'),
                   _play_done_sound=mock.Mock())
        load_definitions(ROOT / 'gr_tabs/parse_tab.py', {'parse_fb2_wrapper'}, env)
        fake = env['FB2Processor'].return_value
        fake.process_book.return_value = iter([(0, '❌ XML parsing error: fixture')])
        fake.stop_parsing = False
        rows = env['parse_fb2_wrapper']('Book', True, 200, False, False,
                                        False, False, False, False, False,
                                        False, False, False, True)
        first = next(rows)
        self.assertIn('Preparing', first[2])
        env['FB2Processor'].assert_not_called()
        remaining = list(rows)
        self.assertIn('❌', remaining[-1][2])
        env['_play_done_sound'].assert_not_called()


class UploadNavigationTests(unittest.TestCase):
    def test_upload_xml_callback_is_imported(self):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8-sig'))
        imported = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == 'gr_tabs.parse_tab'
            for alias in node.names
        }
        self.assertIn('get_xml_files', imported)

    def setUp(self):
        self.env = load_definitions(ROOT / 'app.py',
            {'toggle_tab', 'toggle_tab_parse', 'select_uploaded_next', 'show_project_tabs'},
            dict(gr=gr))

    def test_success_selects_the_analyze_tab(self):
        # After an upload the book is parsed automatically, so the user lands on
        # ANALYZE to check the split before voiceover. Gradio 6 no longer
        # accepts a component instance as an update: it has to be a dict.
        tabs, state = self.env['select_uploaded_next']('Book', '🎉 All 1 projects parsed!')
        self.assertEqual(tabs.get('selected'), 'analyze')
        self.assertTrue(tabs.get('visible'))
        self.assertEqual(state, 'Book')

    def test_analyze_hands_over_to_voiceover(self):
        import re
        from pathlib import Path as _P
        src = (_P(ROOT) / 'gr_tabs/parse_tab.py').read_text(encoding='utf-8-sig')
        self.assertIn('gr.update(visible=True, selected="tts")', src)
        self.assertNotIn('gr.Tabs(visible=True, selected="tts")', src)

    def test_empty_and_failed_upload_do_not_navigate(self):
        for project, status in [('OldBook', ''), ('', '🎉 done'), ('OldBook', '❌ failed')]:
            result = self.env['select_uploaded_next'](project, status)
            self.assertEqual(result, (gr.update(), gr.update()))

    def test_project_change_does_not_reset_selected_tab(self):
        tabs, state = self.env['show_project_tabs']('Book')
        self.assertNotIn('selected', tabs)
        self.assertTrue(tabs['visible'])
        self.assertEqual(state, 'Book')

    def test_real_gradio_event_graph_attaches_navigation_to_upload_only(self):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8-sig'))
        block = next(n for n in tree.body if isinstance(n, ast.With)
                     and any(isinstance(x, ast.Name) and x.id == 'process_file_btn'
                             for x in ast.walk(n)))
        events = []
        for node in block.body:
            text = ast.unparse(node)
            if isinstance(node, ast.Expr) and (text.startswith('process_file_btn.click(')
                                              or text.startswith('sound_done_upload_cb.change(')):
                events.append(node)
        self.assertEqual(len(events), 2)
        env = dict(self.env)
        with gr.Blocks() as demo:
            for name in ['ab_path', 'load_project_dropdown', 'upload_progress_html',
                         'upload_status_text', 'queue_table', 'queue_counter', 'upload_text_file']:
                env[name] = gr.Textbox()
            env['file_queue_state'] = gr.State([])
            env['ab_state'] = gr.State()
            for name in ['remove_ru_cb', 'bg_mode_cb', 'auto_open_upload_cb', 'sound_done_upload_cb']:
                env[name] = gr.Checkbox()
            env['process_file_btn'] = gr.Button()
            env['parse_df_output'] = gr.Dataframe()
            with gr.Tabs() as tabs:
                with gr.Tab('Cover'):
                    pass
                with gr.Tab('Analyze', id='analyze'):
                    pass
            env['inner_tabs'] = tabs
            def process_file_wrapper(*args):
                yield (gr.update(),) * 8
            env['process_file_wrapper'] = process_file_wrapper
            env['get_xml_files'] = lambda ab: []
            env['AppConfig'] = SimpleNamespace(save_user_settings=lambda data: None)
            exec(compile(ast.Module(body=events, type_ignores=[]), 'upload-events', 'exec'), env)
        deps = demo.config['dependencies']
        nav = next(d for d in deps if d['api_name'] == 'select_uploaded_next')
        parent = next(d for d in deps if d['id'] == nav['trigger_after'])
        self.assertEqual(parent['api_name'], 'process_file_wrapper')
        self.assertFalse(nav['queue'])
        sound = next(d for d in deps if any(t[0] == env['sound_done_upload_cb']._id for t in d['targets']))
        self.assertFalse(any(d['trigger_after'] == sound['id'] for d in deps))


if __name__ == '__main__':
    unittest.main()
