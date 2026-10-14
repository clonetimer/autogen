"""Predeclared development-only structural ablations; all are retrained.

Removing branches/differences or untying weights changes parameter counts.
These are diagnostic ablations, NOT compute-matched causal-effect estimates.
"""
from __future__ import annotations
import copy
from pathlib import Path
from .data import Dataset
from .engine import fit
from .graph import analyze_graph, graph_identity
from .io import read_json, write_json, verify, digest, seal
from .models import Shape
from .synthesis import lower_genome, validate_request, training_config


def ablation_genomes(genes: dict) -> list[tuple[str, dict]]:
    result = []
    if any(b['difference'] for b in genes['branches']):
        g = copy.deepcopy(genes)
        for b in g['branches']: b['difference'] = False
        result.append(('no_observation_differences', g))
    if any(b['share'] and b['source'] == 'components' for b in genes['branches']):
        g = copy.deepcopy(genes)
        for b in g['branches']: b['share'] = False
        result.append(('untied_component_weights', g))
    if len(genes['branches']) > 1:
        for i in range(len(genes['branches'])):
            g = copy.deepcopy(genes)
            g['branches'].pop(i)
            g['fusion_order'] = [n if n < i else n-1 for n in genes['fusion_order'] if n != i]
            result.append((f'remove_branch_{i}', g))
    return result


def ablate(dataset_path: str | Path, design_root: str | Path, output: str | Path) -> dict:
    design_root, output = Path(design_root), Path(output)
    if output.exists(): raise FileExistsError('ABLATION_OUTPUT_EXISTS')
    verify(design_root)
    design_plan = read_json(design_root / 'design_plan.json')
    design_report = read_json(design_root / 'design_report.json')
    request = design_plan['request']
    dataset = Dataset(dataset_path, allowed_splits={'train','validation'})
    if dataset.hash != design_plan['dataset_hash']:
        raise ValueError('ABLATION_DEVELOPMENT_DATA_CHANGED')
    winner = design_report['best_composed']
    if winner is None: raise ValueError('NO_DESIGNED_MODEL_TO_ABLATE')
    groups = validate_request(request,dataset.contract)
    shape = Shape(request['window'],len(dataset.contract['channels']),len(dataset.contract['classes']))
    specs, provenance, skipped = [], {}, []
    for name, genes in ablation_genomes(winner['genes']):
        graph = lower_genome(genes,groups,shape)
        if graph_identity(graph) == winner['graph_sha256']:
            skipped.append({'intervention':name,'reason':'NO_EXECUTABLE_CHANGE'})
            continue
        spec={'family':'composed_dag','parameters':graph}
        if digest(spec) in provenance:
            skipped.append({'intervention':name,'reason':'DUPLICATE_INTERVENTION'})
            continue
        specs.append(spec)
        provenance[digest(spec)]={'intervention':name,'genes':genes,'parameters':analyze_graph(shape,graph)['parameter_count']}
    output.mkdir(parents=True)
    write_json(output/'ablation_plan.json',{'selected_graph':winner['graph_sha256'],'source_design':design_report['fingerprint'],
               'interventions':provenance,'skipped':skipped,'test_read':False,'parameter_matched':False,
               'selection_use':'development error analysis only; original champion remains frozen'})
    cfg=training_config(request,specs)
    rows=[]
    if specs:
        write_json(output/'search.json',cfg)
        report=fit(dataset_path,output/'search.json',output/'run')
        for item in report['candidate_results']:
            rows.append({**provenance[digest(item['spec'])], 'status':item['status'],
                         'mean_validation_macro_f1':item.get('mean_validation_macro_f1'),
                         'seed_std_macro_f1':item.get('seed_std_macro_f1')})
    report={'schema':'phm.structural-ablations.v1','status':'COMPLETED' if rows else 'NO_APPLICABLE_ABLATIONS',
            'reference_mean_validation_macro_f1':winner['mean_validation_macro_f1'],
            'reference_parameters':winner['description']['parameter_count'],
            'results':rows,'test_read':False,'champion_changed':False,'parameter_matched':False,
            'claim_status':'DEVELOPMENT_DIAGNOSTICS_ONLY'}
    write_json(output/'ablation_report.json',report)
    seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name not in {'.lock','manifest.json'}])
    return report
