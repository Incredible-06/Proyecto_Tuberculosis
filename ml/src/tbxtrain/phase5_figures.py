"""Figuras para informe desde historiales y predicciones de validation.

Solo lee experimentos previos; todas las exportaciones van a una carpeta nueva.
No carga modelos/imágenes ni escribe en los experimentos de entrenamiento.
"""
import json
from pathlib import Path
import numpy as np
from tbxdata.common import read_csv, sha256, write_csv
from .metrics import binary_curve
from .phase5 import CLASSES, validation_rows, read_validation_predictions
from .protocol import atomic_json

NAMES = {'convnext_tiny':'ConvNeXt-Tiny', 'vmamba_tiny':'VMamba-Tiny',
         'hybrid_cross_attention':'Híbrido de atención cruzada'}
COLORS = ['#2864AD','#C16B19','#287C60']


def _pr_curve(y, score):
    order = np.argsort(-score, kind='stable')
    target, sorted_score = np.asarray(y)[order], np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(target)-1]
    tp = np.cumsum(target)[ends]
    return np.r_[0., tp/target.sum()], np.r_[1.,tp/(ends+1)]


def export_validation_figures(project, base, specs, output, detector_run_id=None, source_files=None):
    import matplotlib.pyplot as plt
    project, base, output = map(Path,(project,base,output))
    rows = validation_rows(project)
    output.mkdir(parents=True,exist_ok=False)
    sources, records, data = {}, [], []
    source_files = source_files or {}

    def source(path):
        path = Path(path)
        sources[str(path)] = sha256(path)
        return path

    source(project/'manifests/validation.csv')

    def save(fig, name, title, table, note, architectures):
        fig.tight_layout()
        try:
            for extension in ['png','pdf','svg']:
                fig.savefig(output/(name+'.'+extension),dpi=300,bbox_inches='tight',facecolor='white')
        finally:
            plt.close(fig)
        write_csv(output/(name+'.csv'),table)
        records.append(dict(name=name,title=title,split='validation',architectures=architectures,
            note=note,files=[name+'.'+extension for extension in ['png','pdf','svg','csv']]))

    with plt.rc_context({'font.size':10,'axes.titlesize':12,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':False,'svg.fonttype':'none'}):
        for spec in specs:
            name = spec['architecture']
            folder = base/'experiments'/spec['run_id']
            paths = source_files.get(spec['run_id'],{})
            summary = json.loads(source(paths.get('summary',folder/'summary.json')).read_text())
            if summary['architecture'] != name or summary['run_id'] != spec['run_id']:
                raise ValueError('Arquitectura/corrida inconsistente.')
            history = read_csv(source(paths.get('history',folder/'history.csv')))
            predictions, y, p, metrics = read_validation_predictions(source(paths.get('predictions',folder/'validation_predictions.csv')),rows)
            if metrics['confusion_matrix'] != summary['validation']['confusion_matrix']:
                raise ValueError('Matriz histórica distinta de las predicciones.')
            data.append((spec,summary,y,p,metrics))
            epochs = [int(r['epoch']) for r in history]
            if len(epochs) != summary['completed_epochs'] or summary['best_epoch'] not in epochs:
                raise ValueError('Historial incompleto.')
            fig,axes = plt.subplots(1,3,figsize=(14,4.3))
            groups = [[('train_loss','Train'),('validation_loss','Validation')],
                      [('train_accuracy','Train'),('validation_accuracy','Validation')],
                      [('validation_macro_f1','Macro-F1'),('validation_tb_recall','Sensibilidad TB')]]
            for ax,columns,title in zip(axes,groups,['Pérdida CE ponderada','Accuracy','Métricas de validation']):
                for color,(key,label) in zip(COLORS,columns):
                    ax.plot(epochs,[float(r[key]) for r in history],marker='o',markersize=3,color=color,label=label)
                ax.axvline(summary['best_epoch'],color='#656565',ls='--',lw=1,label='Mejor época')
                ax.set(title=title,xlabel='Época'); ax.grid(alpha=.18); ax.legend(fontsize=8)
                ax.set_xticks(epochs)
                if ax is not axes[0]: ax.set_ylim(0,1.02)
            fig.suptitle(NAMES[name]+' · historial de entrenamiento',fontsize=14)
            save(fig,name+'_training','Historial de '+NAMES[name],history,
                 'Epochs reales; línea en mejor época. Train se mide durante updates con aumentos; '
                 'validation sin aumentos. No se rellenan épocas tras early stopping.',[name])

            cm = np.asarray(metrics['confusion_matrix'])
            normalized = cm/cm.sum(1,keepdims=True)
            fig,axes = plt.subplots(1,2,figsize=(10,4.7))
            for ax,matrix,title,maximum in zip(axes,[cm,normalized],['Conteos','Normalizada por clase real'],[cm.max(),1.]):
                picture = ax.imshow(matrix,cmap='Blues',vmin=0,vmax=maximum)
                for (i,j),v in np.ndenumerate(matrix):
                    ax.text(j,i,str(int(v)) if matrix is cm else f'{v:.1%}',ha='center',va='center',
                            color='white' if v > maximum*.55 else '#152536')
                ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,
                       xlabel='Clase predicha (argmax)',ylabel='Clase real',title=title)
                fig.colorbar(picture,ax=ax,fraction=.045,pad=.04)
            fig.suptitle(NAMES[name]+' · validation · mejor época '+str(summary['best_epoch']),fontsize=13)
            table = [dict(actual=CLASSES[i],predicted=CLASSES[j],count=int(cm[i,j]),
                          proportion_within_actual=float(normalized[i,j])) for i in range(3) for j in range(3)]
            save(fig,name+'_confusion','Matriz de confusión de '+NAMES[name],table,
                 'Filas reales, columnas predichas; argmax original; normalización por fila.',[name])

            fig,ax = plt.subplots(figsize=(6.1,5))
            table = []
            for i,(cls,color) in enumerate(zip(CLASSES,COLORS)):
                curve = binary_curve(y==i,p[:,i])
                ax.plot(curve['fpr'],curve['tpr'],color=color,label=f"{cls} · AUC={curve['roc_auc']:.4f}")
                table += [dict(class_name=cls,fpr=x,tpr=z) for x,z in zip(curve['fpr'],curve['tpr'])]
            ax.plot([0,1],[0,1],ls='--',color='gray',lw=1)
            ax.set(xlabel='Tasa de falsos positivos',ylabel='Sensibilidad',title=NAMES[name]+'\nROC OVR · validation',
                   xlim=(0,1),ylim=(0,1.01)); ax.legend(loc='lower right'); ax.grid(alpha=.18)
            save(fig,name+'_roc','ROC multiclase de '+NAMES[name],table,
                 'One-vs-rest, scores originales; no corresponde a test.',[name])

        comparison = []
        for spec,s,y,p,m in data:
            comparison.append(dict(architecture=spec['architecture'],run_id=spec['run_id'],best_epoch=s['best_epoch'],
                accuracy=m['accuracy'],macro_f1=m['macro_f1'],TB_recall=m['per_class']['TB']['recall'],
                TB_specificity=m['per_class']['TB']['specificity'],TB_precision=m['per_class']['TB']['precision'],
                TB_roc_auc=m['per_class']['TB']['roc_auc']))
        fig,axes = plt.subplots(1,2,figsize=(12,4.8))
        for ax,columns,title in zip(axes,[['accuracy','macro_f1'],['TB_recall','TB_precision','TB_specificity']],
                                   ['Clasificación multiclase','Clase TB']):
            labels = {'accuracy':'Accuracy','macro_f1':'Macro-F1','TB_recall':'Sensibilidad',
                      'TB_precision':'Precisión','TB_specificity':'Especificidad'}
            x = np.arange(len(comparison)); width = .8/len(columns)
            for j,key in enumerate(columns):
                values = [r[key] for r in comparison]
                bars = ax.bar(x+(j-(len(columns)-1)/2)*width,values,width,label=labels[key],color=COLORS[j])
                ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=8)
            ax.set(xticks=x,xticklabels=['ConvNeXt','VMamba','Híbrido'],ylim=(0,1.16),title=title,ylabel='Proporción')
            ax.legend(loc='lower center',bbox_to_anchor=(.5,-.25),ncol=len(columns),fontsize=9)
            ax.yaxis.grid(alpha=.18); ax.set_axisbelow(True)
        fig.suptitle('Comparación de clasificadores · validation · mejores checkpoints',fontsize=14)
        save(fig,'classifier_comparison','Comparación de los tres clasificadores',comparison,
             'Una seed y validation reutilizada para selección; sin intervalos ni prueba de superioridad estadística.',
             [s['architecture'] for s in specs])

        fig,axes = plt.subplots(1,2,figsize=(12,4.8)); table = []
        for color,(spec,s,y,p,m) in zip(COLORS,data):
            name = spec['architecture']; target = y==2; curve = binary_curve(target,p[:,2])
            recall,precision = _pr_curve(target,p[:,2])
            axes[0].plot(curve['fpr'],curve['tpr'],color=color,
                         label=NAMES[name]+f" · AUC={curve['roc_auc']:.4f}")
            axes[1].step(recall,precision,where='pre',color=color,
                         label=NAMES[name]+f" · AP={curve['average_precision']:.4f}")
            table += [dict(architecture=name,curve='ROC',x=x,y=z) for x,z in zip(curve['fpr'],curve['tpr'])]
            table += [dict(architecture=name,curve='PR',x=x,y=z) for x,z in zip(recall,precision)]
        axes[0].plot([0,1],[0,1],'--',color='gray',lw=1)
        axes[1].axhline(float((data[0][2]==2).mean()),ls='--',color='gray',lw=1,label='Prevalencia de validation')
        for ax,title,xlabel,ylabel in zip(axes,['ROC TB vs resto','Precisión–recall TB'],
                                         ['Tasa de falsos positivos','Sensibilidad'],['Sensibilidad','Precisión']):
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.02))
            ax.legend(fontsize=8,loc='lower left'); ax.grid(alpha=.18)
        fig.suptitle('Comparación de scores TB · validation',fontsize=14)
        save(fig,'TB_roc_precision_recall','ROC y precisión–recall TB comparativas',table,
             'AP es average precision, no área trapezoidal. Scores TB originales; ningún umbral queda seleccionado.',
             [s['architecture'] for s in specs])

        if detector_run_id is not None:
            folder = base/'experiments'/detector_run_id
            paths = source_files.get(detector_run_id,{})
            history = [{k.strip():v for k,v in r.items()} for r in read_csv(source(paths.get('history',folder/'training/results.csv')))]
            detector_summary = json.loads(source(paths.get('summary',folder/'detector_summary.json')).read_text())
            epochs = [int(r['epoch']) for r in history]
            best = max(history,key=lambda r:float(r['metrics/mAP50-95(B)']))
            best_epoch = int(best['epoch'])
            fig,axes = plt.subplots(2,3,figsize=(14,8))
            for ax,key,title in zip(axes[0],['box_loss','cls_loss','dfl_loss'],
                                   ['Pérdida de cajas','Pérdida de clasificación','Pérdida DFL']):
                for prefix,color,label in zip(['train/','val/'],COLORS,['Train','Validation']):
                    ax.plot(epochs,[float(r[prefix+key]) for r in history],color=color,label=label)
                ax.set(title=title,xlabel='Época'); ax.legend(fontsize=8)
            for key,label,color in zip(['metrics/mAP50(B)','metrics/mAP50-95(B)'],['mAP50','mAP50–95'],COLORS):
                axes[1,0].plot(epochs,[float(r[key]) for r in history],color=color,label=label)
            axes[1,0].set(title='mAP por época · validation',xlabel='Época',ylim=(0,1)); axes[1,0].legend(fontsize=8)
            for key,label,color in zip(['metrics/precision(B)','metrics/recall(B)'],['Precisión por caja','Recall por caja'],COLORS):
                axes[1,1].plot(epochs,[float(r[key]) for r in history],color=color,label=label)
            axes[1,1].set(title='P/R por caja · historial',xlabel='Época',ylim=(0,1)); axes[1,1].legend(fontsize=8)
            axes[1,2].axis('off')
            axes[1,2].text(0,1,'YOLO11n · baseline\n\n'+f'{len(history)} épocas ejecutadas\n'+
                f'Mejor mAP50–95 del historial: época {best_epoch}\n\n'+
                'Validación final de best.pt (resumen):\n'+
                f"mAP50 = {detector_summary['map50']:.4f}\n"+
                f"mAP50–95 = {detector_summary['map50_95']:.4f}\n\n"+
                'P/R: punto de curva elegido por Ultralytics.\nNo es sensibilidad por imagen.\nNo define el umbral operativo.',
                va='top',fontsize=10,linespacing=1.5)
            for ax in list(axes[0])+list(axes[1,:2]):
                ax.axvline(best_epoch,color='#656565',ls='--',lw=1); ax.grid(alpha=.18)
            fig.suptitle('Detector YOLO11n · historial de entrenamiento y validation',fontsize=14)
            save(fig,'detector_training','Entrenamiento del detector YOLO11n',history,
                 'La línea marca el mejor mAP50–95 del historial. El resultado final de best.pt procede '
                 'del resumen y puede diferir del CSV de entrenamiento. P/R no son sensibilidad por imagen.', ['yolo11n'])

    if any(sha256(path) != digest for path,digest in sources.items()):
        raise ValueError('Una fuente cambió durante la exportación; descartar esta exportación incompleta.')
    files = {p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    atomic_json(output/'figures_manifest.json',dict(scope='existing_training_and_validation_artifacts',
        test_evaluated=False,external_evaluated=False,new_training_executed=False,figures=records,
        source_sha256=sources,output_sha256=files))
    lines = ['# Figuras para informe — historiales y validation','',
             'PNG a 300 dpi; PDF/SVG vectoriales; CSV de datos por figura. Sin inferencia nueva ni acceso a test.',
             'Las comparaciones usan una seed y validation reutilizada. No incluyen intervalos de confianza.',
             'Las matrices conservan argmax softmax, filas reales y columnas predichas.','']
    for record in records:
        lines += ['## '+record['title'],'',record['note'],'',
                  ' | '.join('['+extension+']('+record['name']+'.'+extension+')' for extension in ['png','pdf','svg','csv']),'']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    return output/'figures_manifest.json'
