from pathlib import Path
import importlib.util
import zipfile
import pytest
S=importlib.util.spec_from_file_location('caas_stage',Path(__file__).parents[2]/'scripts/caas_stage.py')
M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

def archive(path,name='a.txt',data=b'valid'):
    with zipfile.ZipFile(path,'w') as z:z.writestr(name,data)
    return path

def test_probe_roundtrip(tmp_path):
    r=M.probe(tmp_path); assert r['file_roundtrip'] and r['calculation']==45
    assert r['pre_execution_hydration_controlled'] is False

def test_stage_receipt_and_duplicate_commit(tmp_path):
    p=archive(tmp_path/'a.zip');d=tmp_path/'stage'
    r=M.stage(p,d,M.sha(p));assert (d/'a.txt').read_bytes()==b'valid'
    assert r['expanded_bytes']==5
    with pytest.raises(FileExistsError):M.stage(p,d,M.sha(p))

@pytest.mark.parametrize('name',['../escape','/absolute','a/../../bad','back\\slash','C:/bad'])
def test_zip_slip_blocked(tmp_path,name):
    p=archive(tmp_path/'a.zip',name);d=tmp_path/'stage'
    with pytest.raises(ValueError):M.stage(p,d,M.sha(p))
    assert not d.exists() and not list(tmp_path.glob('stage.partial-*'))

def test_hash_and_expansion_budget(tmp_path):
    p=archive(tmp_path/'a.zip',data=b'x'*100)
    with pytest.raises(ValueError,match='SHA256'):M.stage(p,tmp_path/'a','0'*64)
    with pytest.raises(ValueError,match='EXPANSION'):M.stage(p,tmp_path/'a',M.sha(p),max_expanded_bytes=10)

def test_symlink(tmp_path):
    p=tmp_path/'a.zip';i=zipfile.ZipInfo('link');i.create_system=3;i.external_attr=0o120777<<16
    with zipfile.ZipFile(p,'w') as z:z.writestr(i,'/etc/passwd')
    with pytest.raises(ValueError,match='SPECIAL'):M.stage(p,tmp_path/'a',M.sha(p))

def test_nested_zip_is_not_extracted(tmp_path):
    nested=archive(tmp_path/'nested.zip');outer=archive(tmp_path/'outer.zip','inner.zip',nested.read_bytes())
    M.stage(outer,tmp_path/'stage',M.sha(outer));assert (tmp_path/'stage/inner.zip').is_file()
    assert not (tmp_path/'stage/a.txt').exists()
