"""Resúmenes capturables sin mezclar nuevas celdas con evidencia histórica."""
import textwrap


def evidence(cells, identifier, title, explanation, source):
    cells.append({'cell_type': 'markdown', 'id': identifier + '-title',
                  'metadata': {'tags': ['evidencia', identifier]},
                  'source': (f'## {identifier} {title}\n\n' + explanation +
                             '\n\n**Captura:** incluya este título y toda la salida de la celda siguiente.\n').splitlines(keepends=True)})
    cells.append({'cell_type': 'code', 'id': identifier,
                  'metadata': {'tags': ['evidencia', identifier]},
                  'execution_count': None, 'outputs': [],
                  'source': textwrap.dedent(source).strip().splitlines(keepends=True)})


def refine_environment(nb):
    cells = nb['cells']
    # El último texto se sustituye por un cierre vigente.
    cells[-1]['source'] = ['## Cierre de la fase de entorno\n',
        '\nLa ejecución histórica de Colab verificó A100 y 12.278 hashes completos. '
        'Las celdas siguientes generan resúmenes de ESTA ejecución. En CPU el estado '
        'CUDA debe quedar pendiente; no se reutiliza un resultado anterior como si fuera nuevo.\n']
    evidence(cells, 'F1-E1', 'Entorno y prueba CUDA',
             'Una multiplicación pequeña verifica el funcionamiento básico de CUDA. La compatibilidad de los modelos se comprobará en la etapa de entrenamiento.', '''
        print('FASE 1 | ENTORNO DE ESTA EJECUCIÓN')
        print('Colab:', setup_report['in_colab'], '| Python:', setup_report['python'])
        print('PyTorch:', setup_report.get('torch', 'no revisado localmente'))
        print('GPU:', setup_report.get('gpu', {}).get('name', 'no disponible'))
        print('CUDA build:', setup_report.get('cuda_build', 'no disponible'))
        print('Prueba CUDA:', setup_report['cuda_probe']['status'])
        print('Seed:', setup_report['seed'], '| Entrenamiento:', setup_report['training_executed'])
        print('Operación:', setup_report['cuda_probe'].get('operation', 'no ejecutada'))
    ''')
    evidence(cells, 'F1-E2', 'Transferencia y persistencia',
             'Los hashes comprueban la transferencia contra el inventario previo. Las carpetas experiments y checkpoints pueden estar vacías porque todavía no hay entrenamientos.', '''
        print('FASE 1 | INTEGRIDAD Y PERSISTENCIA')
        print('Integridad:', transfer_report['status'])
        print('SHA-256 completos comprobados:', transfer_report['all_image_hashes_verified'])
        print('Cruces internos conocidos:', sum(transfer_report['overlap_checks'].values()))
        print('Resultados:', setup_output)
        print('Carpetas futuras:', setup_report.get('output_paths', 'solo se crean en Drive desde Colab'))
        print('No se generan pesos ni checkpoints en estas fases.')
        if not IN_COLAB or setup_report['cuda_probe']['status'] != 'passed':
            print('CIERRE GPU PENDIENTE EN ESTA EJECUCIÓN')
    ''')


def refine_data(nb):
    cells = nb['cells']
    evidence(cells, 'F2-E1', 'Categorías y decisión de clasificación',
             'Se estudian las cuatro categorías descriptivas del conjunto. El clasificador inicial agrupa los subtipos de TB; no convierte una etiqueta de lesión en una clase exclusiva de imagen.', '''
        from collections import Counter
        print('FASE 2 | DATOS Y CATEGORÍAS')
        print('Imágenes auditadas:', audit['images'], '| Tamaño:', audit['dimensions'])
        print('Base etiquetada:', manifest_meta['original_labeled_files'])
        print('Subtipos por imagen TB:', audit['xml_image_category_sets'])
        print('Clases del clasificador:', config['classes'])
        print('30 casos mixtos y 1 TB sin cajas: se conserva la clase TB.')
        print('No se descartan subtipos: permanecen en los metadatos de lesiones.')
        print('Montgomery/Shenzhen: 400 imágenes combinadas reservadas; no son las fuentes completas.')
    ''')
    evidence(cells, 'F2-E2', 'Particiones y control de fuga',
             'La ausencia de cruces corresponde a rutas, hashes y grupos conocidos. No demuestra independencia por paciente.', '''
        print('FASE 2 | PARTICIONES CONGELADAS | seed', config['seed'])
        print(f"{'Partición':<14}{'Health':>8}{'Sick':>8}{'TB':>8}{'Total':>9}")
        for split, values in manifest_meta['counts'].items():
            print(f"{split:<14}{values['health']:>8}{values['sick']:>8}{values['TB']:>8}{sum(values.values()):>9}")
        print('Contenidos únicos:', integrity['supervised_images'], '| Grupos:', integrity['supervised_groups'])
        for field in ['relative_path', 'sha256', 'rgb_sha256', 'group_id']:
            values = [v for k, v in integrity['overlap_checks'].items() if k.startswith(field + ':')]
            assert len(values) == 3 and values == [0, 0, 0]
            print(field, 'train/val | train/test | val/test:', *values)
        print('Integridad:', integrity['status'], '| Hashes completos:', integrity['all_image_hashes_verified'])
        print('Independencia por paciente garantizada:', integrity['patient_independence_guaranteed'])
    ''')
    evidence(cells, 'F2-E3', 'Muestras de entrenamiento y cajas',
             'Todas las muestras pertenecen a train. Rojo identifica ActiveTuberculosis y cian ObsoletePulmonaryTuberculosis. Son anotaciones originales, no predicciones.', '''
        show_figure(FIGURES / 'training_samples.png')
    ''')
    evidence(cells, 'F2-E4', 'Preprocesamiento y aumentos',
             'Las variantes se muestran antes de normalizar. Brillo y contraste ±8 %, rotación ±5°; aumentos solo en train. CLAHE desactivado.', '''
        print('Salida de clasificación: float32, 3 × 512 × 512')
        print('Media:', config['normalization']['mean'], '| Desviación:', config['normalization']['std'])
        print('Aumentos solo en train:', config['augmentation'])
        show_figure(FIGURES / 'training_transformations.png')
    ''')
    evidence(cells, 'F2-E5', 'Lector verificado y estructura de detección',
             'El lector es de clasificación. Las cajas quedan preparadas como metadatos originales; su adaptación al detector y su aumento geométrico pertenecen a la etapa de detección.', '''
        print('FASE 2 | COMPROBACIÓN FUNCIONAL')
        print('Muestras verificadas:', len(checked_samples), '| Train y validación repetibles')
        print('Lote PyTorch CPU:', dataloader_check)
        print('Cajas COCO preservadas:', sum(int(r['bbox_count']) for rows in
              [read_csv(PROJECT_ROOT / 'manifests' / f'{s}.csv') for s in ['train', 'validation', 'test']]
              for r in rows))
        missing = next(train_ds.metadata(i) for i, row in enumerate(train_ds.rows) if row['image_id'] == 'tb1199')
        print('tb1199:', missing['annotation_status'], '| No usar como negativo del detector')
        print('Espacio de cajas:', missing['box_coordinate_space'])
        print('Exportación de datos:', drive_result if IN_COLAB else RUN_DIR)
        print('Entrenamiento ejecutado: False')
    ''')
    evidence(cells, 'F2-E6', 'Fuentes externas completas y reservadas',
             'Resumen de la preparación registrada de Montgomery y Shenzhen. Se contrasta el hash de sus manifiestos. Esta celda no descarga ni afirma haber comprobado las imágenes externas en la sesión Colab actual; los originales externos se conservan aparte.', '''
        external_summary = json.loads((PROJECT_ROOT / 'reports/external_sources/preparation_summary.json').read_text(encoding='utf-8'))
        print('FASE 2 | FUENTES EXTERNAS OFICIALES')
        print('Estado registrado:', external_summary['status'])
        for name, values in external_summary['datasets'].items():
            print(name, '| Imágenes:', values['images'], '| Normal:', values['normal'],
                  '| Anormal compatible con TB:', values['abnormal_tb_consistent'],
                  '| Lecturas:', values['clinical_readings'])
        for name, expected in external_summary['outputs_sha256'].items():
            assert sha256(PROJECT_ROOT / 'manifests/external' / name) == expected, name
        print('Manifiestos externos: hashes coincidentes con la preparación registrada')
        print('Excluidos de train/validation/test de TBX11K:', external_summary['excluded_from_tbx11k_splits'])
        print('Evaluación independiente habilitada:', external_summary['eligible_for_evaluation'])
        print('No se equiparan sus etiquetas binarias a las tres clases internas.')
    ''')
    nb['metadata']['revision'] = {'date': '2026-10-21', 'academic_phase': 2,
        'historical_outputs_preserved_separately': True}
