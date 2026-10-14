import copy
from phm_design.reference_design import portfolio_backend
from phm_design.synthesis import propose
from phm_design.optuna_proposal import propose_optuna_tpe
from tests.phm_design.test_optimizer_multigeneration_dev30 import _req,_contract,_rec


def test_portfolio_schedule_fixed_and_equal():
    assert [portfolio_backend(i) for i in range(5)] == ["feedback","feedback","optuna_tpe","feedback","optuna_tpe"]


def test_portfolio_keeps_shared_startup_and_unique_topologies():
    r=_req();c=_contract();seen=set();archive=[];parents=[];top=[];backends=[]
    for g in range(5):
        backend=portfolio_backend(g);backends.append(backend)
        if backend=="feedback":
            out=propose(r,c,g,parents,seen)
        else:
            out=propose_optuna_tpe(r,c,g,archive,seen)
        assert len(out["accepted"])==(3 if g==0 else 1)
        for i,p in enumerate(out["accepted"]):
            assert p["graph_sha256"] not in seen
            seen.add(p["graph_sha256"]);top.append(p["topology_sha256"])
            rec=_rec(p,.70+.03*g+.001*i,g,i)
            archive.append(rec);parents.append(rec)
    assert backends == ["feedback","feedback","optuna_tpe","feedback","optuna_tpe"]
    assert len(top)==7 and len(set(top))==7


def test_portfolio_generation_zero_matches_native_and_optuna():
    r=_req();c=_contract()
    a=propose(r,c,0,[],set());b=propose_optuna_tpe(r,c,0,[],set())
    assert [x["graph_sha256"] for x in a["accepted"]]==[x["graph_sha256"] for x in b["accepted"]]
