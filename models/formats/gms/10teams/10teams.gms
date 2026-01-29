* MPS Input File = ../../mps/10teams/10teams.mps

Sets i        "all rows in MPS order"
     ig(i)    "greater-than-or equal rows"
     il(i)    "less-than-or equal rows"
     ie(i)    "equality rows"
     ir(i)    "ranged rows"

     j        "all columns in MPS order"
     jc (j)   "continuous columns"
     jb (j)   "binary columns"
     ji (j)   "integer columns"
     jsc(j)   "semi-continuous columns"
     jsi(j)   "semi-integer columns"
     s        "sos sets"
     js1(s,j) "sos 1 columns"
     js2(s,j) "sos 2 columns"

     e        "equation name stems"
     v        "variable name stems"
     ei(e,i)  "equation type mapping"
     js(j,s)  "SOS mapping";

Alias (j,jj), (v,vv), (js1,jss1), (js2,jss2), (s,ss);

Parameters  c(j)        "objective coefficients"
            cobj        "objective constant"
            b(i)        "right hand sides"
            ac (i,jc)   "matrix coefficients: continuous variables"
            ab (i,jb)   "matrix coefficients: binary variables"
            ai (i,ji)   "matrix coefficients: integer variables"
            asc(i,jsc)  "matrix coefficients: semi-continuous variables"
            asi(i,jsi)  "matrix coefficients: semi-integer variables"
            as1(i,s,j)  "matrix coefficients: sos 1 variables"
            as2(i,s,j)  "matrix coefficients: sos 2 variables";

Parameters  qobj(    v,jj,vv,jj) "Q term for objective"
            q   (e,i,v,jj,vv,jj) "Q terms by equation type";

Set mps2gms;
Parameter mps2gmsstats(mps2gms);

Equations eobj   "objective function"
          eg(i)  "greater-than-or equal equations"
          el(i)  "less-than-or equal equations"
          ee(i)  "equality equations"
          er(i)  "ranged equations";

Free     Variable obj        "objective variable";
Positive Variable xc (j)     "continuous variables";
Positive Variable r  (i)     "ranged row variables";
Binary   Variable xb (j)     "binary variables";
Integer  Variable xi (j)     "integer variables";
Semicont Variable xsc(j)     "semi-continuous variables";
Semiint  Variable xsi(j)     "semi-integer variables"
SOS1     Variable xs1(s,j)   "SOS 1 variables"
SOS2     Variable xs2(s,j)   "SOS 2 variables";

$GDXin ../../gdx/10teams/10teams.gdx
$Load i j mps2gms s mps2gmsstats
$Load ig il ie ir
$Load jc jb ji jsc jsi js1 js2
$Load cobj c b
$Load ac ab ai asc asi as1 as2
$Load xc xb xi xsc xsi xs1 xs2 r
$Load e v ei js qobj q
$GDXin

eobj.. sum(jc,       c(jc) *xc (jc))
     + sum(jb,       c(jb) *xb (jb))
     + sum(ji,       c(ji) *xi (ji))
     + sum(jsc,      c(jsc)*xsc(jsc))
     + sum(jsi,      c(jsi)*xsi(jsi))
     + sum(js1(s,j), c(j)  *xs1(js1))
     + sum(js2(s,j), c(j)  *xs2(js2))
     + cobj
     =e= obj;

eg(ig).. sum(jc, ac(ig,jc )*xc (jc ))
     + sum(jb,  ab (ig,jb )*xb (jb ))
     + sum(ji,  ai (ig,ji )*xi (ji ))
     + sum(jsc, asc(ig,jsc)*xsc(jsc))
     + sum(jsi, asi(ig,jsi)*xsi(jsi))
     + sum(js1, as1(ig,js1)*xs1(js1))
     + sum(js2, as2(ig,js2)*xs2(js2))
     =g= b(ig);

el(il).. sum(jc, ac(il,jc )*xc (jc ))
     + sum(jb,  ab (il,jb )*xb (jb ))
     + sum(ji,  ai (il,ji )*xi (ji ))
     + sum(jsc, asc(il,jsc)*xsc(jsc))
     + sum(jsi, asi(il,jsi)*xsi(jsi))
     + sum(js1, as1(il,js1)*xs1(js1))
     + sum(js2, as2(il,js2)*xs2(js2))
     =l= b(il);

ee(ie).. sum(jc, ac(ie,jc )*xc (jc ))
     + sum(jb,  ab (ie,jb )*xb (jb ))
     + sum(ji,  ai (ie,ji )*xi (ji ))
     + sum(jsc, asc(ie,jsc)*xsc(jsc))
     + sum(jsi, asi(ie,jsi)*xsi(jsi))
     + sum(js1, as1(ie,js1)*xs1(js1))
     + sum(js2, as2(ie,js2)*xs2(js2))
     =e= b(ie);

er(ir).. sum(jc, ac(ir,jc )*xc (jc ))
     + sum(jb,  ab (ir,jb )*xb (jb ))
     + sum(ji,  ai (ir,ji )*xi (ji ))
     + sum(jsc, asc(ir,jsc)*xsc(jsc))
     + sum(jsi, asi(ir,jsi)*xsi(jsi))
     + sum(js1, as1(ir,js1)*xs1(js1))
     + sum(js2, as2(ir,js2)*xs2(js2))
     =e= r(ir);

Model m / all /;

Option limcol=0, limrow=0, solprint=off;

Solve m using mip minimizing obj;
