"""Reproducible arithmetic/counterexample checks supporting direct review.
These checks support specific claims, not completeness or human-gold status.
Run from any directory with Python's standard library only.
"""
import json
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import cache
from itertools import permutations, product
from pathlib import Path

out = {}

@cache
def bob(b, w):
    if b+w == 0:
        return F(0)
    return (F(b,b+w)*(1+alice(b-1,w)) if b else 0)+(F(w,b+w)*alice(b,w-1) if w else 0)

@cache
def alice(b, w):
    if b+w == 0:
        return F(0)
    return bob(b-1,w) if b else bob(0,w-1)

assert bob(3,1) == F(11,8)
assert alice(10,10) == F(45,19)
out['ball_expectations'] = {'B_3_1':str(bob(3,1)), 'Alice_first_Bob_10_10':str(alice(10,10))}

assert 98*7+107+6 == 799
assert F(799,200)+3 == F(1399,200)
assert F(495,1000)*4+F(3535,1000) == F(1103,200)
assert 2013 == 3*671
out['new_arithmetic_findings'] = {'index13_sum':799,'index13_X':'6.995','index15_f4_argument':'5.515','index15_f4_rounded':6}

counts=(2,0,2,0,2,0)
# z^2=z-1 for a primitive sixth root z.
powers=((1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1))
assert tuple(sum(c*p[j] for c,p in zip(counts,powers)) for j in range(2)) == (0,0)
assert counts[0] != counts[3]
assert len(set(permutations(range(6)))) == 720
out['cyclotomic_counterexample'] = {'exponents':[0,2,4,6,8,10], 'residue_counts':counts,'six_distinct_residue_orders':720}

intermediate=[1]+[101]*99
assert sum(intermediate)==10000 and intermediate.count(100)==0
out['candy_counterexample']={'move':'one child gives 1 candy to each of the other 99','children_still_at_100':0}

# Coordinates divided by sqrt(5); use squared scale 1/5.
A,E,B,C,D=(-4,0),(0,0),(-3,3),(9,9),(36,0)
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return a[0]*b[1]-a[1]*b[0]
def tan_angle(a,v,b):
    u,w=sub(a,v),sub(b,v)
    return F(abs(cross(u,w)),dot(u,w))
assert dot(sub(B,E),sub(C,E)) == 0
assert tan_angle(E,A,B)==tan_angle(E,B,C)==3
assert tan_angle(E,D,C)==tan_angle(E,C,B)==F(1,3)
assert F(abs(cross(sub(C,E),sub(D,E))),abs(cross(sub(A,E),sub(B,E))))==27
assert F(dot(sub(B,C),sub(B,C)),5)==36
assert F(dot(sub(A,D),sub(A,D)),5)==320
out['quadrilateral_indices23_25']={'coordinates_divided_by_sqrt5':dict(A=A,E=E,B=B,C=C,D=D),'CED_over_AEB':27,'BC':6,'AD_squared':320}

with localcontext() as ctx:
    ctx.prec=70
    z=Decimal(21).sqrt()
    a=(Decimal(0),Decimal(0));b=(Decimal(16)/5,Decimal(12)/5)
    c=(Decimal(5),Decimal(0));d=(Decimal(21)/5,-2*z/5)
    p=((28+2*z)/5,Decimal(0))
    def refl(v,u):
        k=2*dot(u,v)/dot(u,u)
        return tuple(k*x-y for x,y in zip(u,v))
    for axis,anchor,other in [(b,b,d),(d,d,b)]:
        v=refl(sub(other,anchor),axis)
        assert abs(cross(sub(p,anchor),v))<Decimal('1e-60')
    assert abs(dot(b,b)-16)<Decimal('1e-60')
    assert abs(dot(sub(b,c),sub(b,c))-9)<Decimal('1e-60')
    assert abs(dot(sub(c,d),sub(c,d))-4)<Decimal('1e-60')
    assert abs(dot(d,d)-21)<Decimal('1e-60')
    # B and D lie on the circle with diameter AC.
    assert abs(dot(b,sub(b,c)))<Decimal('1e-60')
    assert abs(dot(d,sub(d,c)))<Decimal('1e-60')
    out['cyclic_indices32_34']={'DA_squared':21,'intersection_P_x_exact':'(28+2*sqrt(21))/5','intersection_P_x':str(p[0]),'P_beyond_C':p[0]>5}
    value=Decimal(3)
    for _ in range(7):
        value=(5*value+3*(value*value-4).sqrt())/4
    assert int(value)==335
    out['recurrence_final_a8']=str(value)

area=F(7225,64)*(F(240,289)+F(5544,7225)+F(24,25))
assert area==F(1155,4)
out['hexagon_area']=str(area)

seqs=[v for v in product(range(1,6),repeat=5) if all(v[j]<=j+1 for j in range(5)) and list(v)==sorted(v)]
assert len(seqs)==42
triples=[(a,b,54000//(a**4*b*b)) for a in range(1,16) for b in range(1,233) if 54000%(a**4*b*b)==0]
assert len(triples)==16
strings={''.join(x) for x in permutations('HMMTHMMT')}
assert len(strings)==420
assert Counter(s.count('HMMT') for s in strings)=={0:361,1:58,2:1}
assert 'HMHMMTMT'[1:5]=='MHMM'
out['enumerations']={'weak_sequences':42,'a4_b2_c':16,'HMMT_free':361,'HMHMMTMT_positions2_to5':'MHMM'}

assert len({7,210,216})==3 and 210%42==216%72==0
assert 216-210<7<216+210
out['triangle_last_problem']={'sides':[7,210,216],'minimum_third_side':7}

if __name__=='__main__':
    path=Path(__file__).with_name('mathematical_checks.json')
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(f'PASS: independent arithmetic and counterexample checks; {path}')
