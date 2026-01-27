# MPS Input File = models/formats/mps/10teams/reduced10teams.mps

import gamspy as gp
import sys

gp.set_options({"USE_PY_VAR_NAME": "yes"})

cont = gp.Container()

i   = gp.Set(cont,             description = "all rows in MPS order")
ig  = gp.Set(cont, domain=[i], description = "greater-than-or equal rows")
il  = gp.Set(cont, domain=[i], description = "less-than-or equal rows")
ie  = gp.Set(cont, domain=[i], description = "equality rows")
ir  = gp.Set(cont, domain=[i], description = "ranged rows")

j   = gp.Set(cont,               description = "all columns in MPS order")
jc  = gp.Set(cont, domain=[j],   description = "continuous columns")
jb  = gp.Set(cont, domain=[j],   description = "binary columns")
ji  = gp.Set(cont, domain=[j],   description = "integer columns")
jsc = gp.Set(cont, domain=[j],   description = "semi-continuous columns")
jsi = gp.Set(cont, domain=[j],   description = "semi-integer columns")
s   = gp.Set(cont,               description = "sos sets")
js1 = gp.Set(cont, domain=[s,j], description = "sos 1 columns")
js2 = gp.Set(cont, domain=[s,j], description = "sos 2 columns")

e   = gp.Set(cont, description = "equation name stems")
v   = gp.Set(cont, description = "variable name stems")
ei  = gp.Set(cont, domain=[e,i], description = "equation type mapping")
js  = gp.Set(cont, domain=[j,s], description = "SOS mapping")

jj   = gp.Alias(cont, alias_with = j)
vv   = gp.Alias(cont, alias_with = v)
jss1 = gp.Alias(cont, alias_with = js1)
jss2 = gp.Alias(cont, alias_with = js2)
ss   = gp.Alias(cont, alias_with = s)

c        = gp.Parameter(cont, domain=[j],   description = "objective coefficients")
cobj     = gp.Parameter(cont,               description = "objective constant")
objsense = gp.Parameter(cont,               description = "objective sense")
b      = gp.Parameter(cont, domain=[i],     description = "right hand sides")
stagei = gp.Parameter(cont, domain=[i],     description = "block numbers of equations in decomposition")
stages = gp.Parameter(cont, domain=[s],     description = "block numbers of SOS in decomposition")
ac     = gp.Parameter(cont, domain=[i,jc],  description = "matrix coefficients: continuous variables")
ab     = gp.Parameter(cont, domain=[i,jb],  description = "matrix coefficients: binary variables")
ai     = gp.Parameter(cont, domain=[i,ji],  description = "matrix coefficients: integer variables")
asc    = gp.Parameter(cont, domain=[i,jsc], description = "matrix coefficients: semi-continuous variables")
asi    = gp.Parameter(cont, domain=[i,jsi], description = "matrix coefficients: semi-integer variables")
as1    = gp.Parameter(cont, domain=[i,s,j], description = "matrix coefficients: sos 1 variables")
as2    = gp.Parameter(cont, domain=[i,s,j], description = "matrix coefficients: sos 2 variables")

qobj   = gp.Parameter(cont, domain=[v,j,v,j],     description = "Q term for objective")
q      = gp.Parameter(cont, domain=[e,i,v,j,v,j], description = "Q terms by equation type")

mps2gms = gp.Set(cont)
mps2gmsstats = gp.Parameter(cont, domain=[mps2gms])

eobj = gp.Equation(cont,             description = "objective function")
eg   = gp.Equation(cont, domain=[i], description = "greater-than-or equal equations")
el   = gp.Equation(cont, domain=[i], description = "less-than-or equal equations")
ee   = gp.Equation(cont, domain=[i], description = "equality equations")
er   = gp.Equation(cont, domain=[i], description = "ranged equations")

obj = gp.Variable(cont, type = "free",                   description = "objective variable")
xc  = gp.Variable(cont, type = "positive", domain=[j],   description = "continuous variables")
r   = gp.Variable(cont, type = "positive", domain=[i],   description = "ranged row variables")
xb  = gp.Variable(cont, type = "binary",   domain=[j],   description = "binary variables")
xi  = gp.Variable(cont, type = "integer",  domain=[j],   description = "integer variables")
xsc = gp.Variable(cont, type = "semicont", domain=[j],   description = "semi-continuous variables")
xsi = gp.Variable(cont, type = "semiint",  domain=[j],   description = "semi-integer variables")
xs1 = gp.Variable(cont, type = "sos1",     domain=[s,j], description = "SOS 1 variables")
xs2 = gp.Variable(cont, type = "sos2",     domain=[s,j], description = "SOS 2 variables")

cont.loadRecordsFromGdx('models/formats/gdx/10teams/reduced10teams.gdx')
#print(mps2gmsstats.records)

eobj[...] = \
       gp.Sum(jc,       c[jc]  * xc [jc])  \
     + gp.Sum(jb,       c[jb]  * xb [jb])  \
     + gp.Sum(ji,       c[ji]  * xi [ji])  \
     + gp.Sum(jsc,      c[jsc] * xsc[jsc]) \
     + gp.Sum(jsi,      c[jsi] * xsi[jsi]) \
     + gp.Sum(js1[s,j], c[j]   * xs1[js1]) \
     + gp.Sum(js2[s,j], c[j]   * xs2[js2]) \
     + cobj == obj

eg[ig] = \
       gp.Sum(jc,  ac [ig,jc ] * xc [jc ]) \
     + gp.Sum(jb,  ab [ig,jb ] * xb [jb ]) \
     + gp.Sum(ji,  ai [ig,ji ] * xi [ji ]) \
     + gp.Sum(jsc, asc[ig,jsc] * xsc[jsc]) \
     + gp.Sum(jsi, asi[ig,jsi] * xsi[jsi]) \
     + gp.Sum(js1, as1[ig,js1] * xs1[js1]) \
     + gp.Sum(js2, as2[ig,js2] * xs2[js2]) \
     >= b[ig]

el[il] = \
       gp.Sum(jc,  ac [il,jc ] * xc [jc ]) \
     + gp.Sum(jb,  ab [il,jb ] * xb [jb ]) \
     + gp.Sum(ji,  ai [il,ji ] * xi [ji ]) \
     + gp.Sum(jsc, asc[il,jsc] * xsc[jsc]) \
     + gp.Sum(jsi, asi[il,jsi] * xsi[jsi]) \
     + gp.Sum(js1, as1[il,js1] * xs1[js1]) \
     + gp.Sum(js2, as2[il,js2] * xs2[js2]) \
     <= b[il]

ee[ie] = \
       gp.Sum(jc,  ac [ie,jc ] * xc [jc ]) \
     + gp.Sum(jb,  ab [ie,jb ] * xb [jb ]) \
     + gp.Sum(ji,  ai [ie,ji ] * xi [ji ]) \
     + gp.Sum(jsc, asc[ie,jsc] * xsc[jsc]) \
     + gp.Sum(jsi, asi[ie,jsi] * xsi[jsi]) \
     + gp.Sum(js1, as1[ie,js1] * xs1[js1]) \
     + gp.Sum(js2, as2[ie,js2] * xs2[js2]) \
     == b[ie]

er[ir] = \
       gp.Sum(jc,  ac [ir,jc ] * xc [jc ]) \
     + gp.Sum(jb,  ab [ir,jb ] * xb [jb ]) \
     + gp.Sum(ji,  ai [ir,ji ] * xi [ji ]) \
     + gp.Sum(jsc, asc[ir,jsc] * xsc[jsc]) \
     + gp.Sum(jsi, asi[ir,jsi] * xsi[jsi]) \
     + gp.Sum(js1, as1[ir,js1] * xs1[js1]) \
     + gp.Sum(js2, as2[ir,js2] * xs2[js2]) \
     == r[ir]

mps_model = gp.Model(cont, problem = "mip", sense = "min", objective = obj, equations = cont.getEquations())
mps_model.solve(output = sys.stdout)
