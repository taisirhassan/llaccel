"""generate matching svg previews and editable draw.io diagrams."""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'assets'
COLORS = {'plain': ('#ffffff', '#52525b'), 'blue': ('#eff6ff', '#2563eb'),
          'green': ('#ecfdf5', '#059669'), 'purple': ('#f5f3ff', '#7c3aed')}

class Diagram:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{name}">',
            '<defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.7" fill="#d4d4d8"/></pattern><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#52525b" stroke-width="1.5"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="white"/><rect width="{width}" height="{height}" fill="url(#grid)"/>', '<g font-family="Helvetica, Arial, sans-serif" fill="#27272a">']
        self.file = ET.Element('mxfile', host='app.diagrams.net')
        page = ET.SubElement(self.file, 'diagram', name=name)
        model = ET.SubElement(page, 'mxGraphModel', grid='1', gridSize='20', page='1', pageWidth=str(width), pageHeight=str(height))
        self.root = ET.SubElement(model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.next_id = 2
    def cell(self, value, style, x, y, w, h):
        ident = str(self.next_id); self.next_id += 1
        c = ET.SubElement(self.root, 'mxCell', id=ident, value=value, style=style, vertex='1', parent='1')
        ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})
        return ident
    def label(self, x, y, value, size=13):
        self.svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="#52525b">{escape(value)}</text>')
        self.cell(value, f'text;html=0;align=left;verticalAlign=middle;fontSize={size};fontColor=#52525b;', x, y-16, len(value)*size*.6, 22)
    def group(self,x,y,w,h,title):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#fafafa" stroke="#a1a1aa" stroke-dasharray="6 5"/>')
        self.cell('', 'rounded=0;fillColor=#fafafa;strokeColor=#a1a1aa;dashed=1;',x,y,w,h)
        self.label(x+16,y+25,title,14)
    def node(self,x,y,w,h,title,detail='',color='plain'):
        fill, stroke = COLORS[color]
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        ty=y+h/2-(4 if detail else -5)
        self.svg.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="16" font-weight="600">{escape(title)}</text>')
        if detail:
            self.svg.append(f'<text x="{x+w/2}" y="{ty+23}" text-anchor="middle" font-size="12" fill="#52525b">{escape(detail)}</text>')
        return self.cell(title+('\n'+detail if detail else ''),f'rounded=1;arcSize=6;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};fontSize=15;fontColor=#27272a;',x,y,w,h)
    def edge(self,points,source=None,target=None,dashed=False,both=False):
        coords=' '.join(('M' if i==0 else 'L')+f' {x} {y}' for i,(x,y) in enumerate(points))
        self.svg.append(f'<path d="{coords}" fill="none" stroke="#52525b" stroke-width="1.5" marker-end="url(#arrow)"'+(' marker-start="url(#arrow)"' if both else '')+(' stroke-dasharray="5 4"' if dashed else '')+'/>')
        ident=str(self.next_id);self.next_id+=1
        attrs=dict(id=ident,edge='1',parent='1',style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=open;endFill=0;strokeColor=#52525b;'+('dashed=1;' if dashed else '')+('startArrow=open;startFill=0;' if both else ''))
        if source: attrs['source']=source
        if target: attrs['target']=target
        c=ET.SubElement(self.root,'mxCell',attrs)
        g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        for key,p in [('sourcePoint',points[0]),('targetPoint',points[-1])]:
            ET.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),**{'as':key})
        a=ET.SubElement(g,'Array',**{'as':'points'})
        for x,y in points[1:-1]: ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
    def save(self):
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/(self.name+'.svg')).write_text('\n'.join(self.svg+['</g></svg>'])+'\n')
        ET.indent(self.file)
        ET.ElementTree(self.file).write(OUT/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)


def architecture():
    d=Diagram('architecture',1040,660)
    d.group(240,40,760,440,'llaccel core')
    host=d.node(40,100,160,70,'host','tokens / embeddings')
    cp=d.node(280,100,190,70,'command processor','queues + semaphores')
    arb=d.node(760,100,200,70,'DRAM arbiter','3 requesters')
    dram=d.node(760,550,200,70,'DRAM','weights / programs / KV','green')
    d.edge([(200,135),(280,135)],host,cp)
    d.edge([(470,135),(760,135)],cp,arb,both=True)
    d.label(548,122,'instruction fetch')
    engines=[]
    for x,title,detail,c in [(280,'matrix','16 × 16 INT8','blue'),(450,'vector','16 INT16 lanes','purple'),(620,'attention','64 MAC lanes','blue'),(790,'DMA','2D transfers','plain')]:
        n=d.node(x,280,150,70,title,detail,c);engines.append(n)
        d.edge([(375,170),(375,230),(x+75,230),(x+75,280)],cp,n,dashed=True)
    d.label(465,216,'instruction issue')
    d.edge([(820,170),(820,195),(775,195),(775,315),(770,315)],arb,engines[2],both=True)
    d.edge([(920,170),(975,170),(975,315),(940,315)],arb,engines[3],both=True)
    xbar=d.node(280,410,660,40,'SRAM crossbar')
    for x,n in zip((355,525,695,865),engines):d.edge([(x,350),(x,410)],n,xbar,both=True)
    sram=d.node(360,550,290,70,'SRAM','1 MiB / 16 banks','green')
    d.edge([(505,450),(505,550)],xbar,sram,both=True)
    d.edge([(960,135),(1020,135),(1020,585),(960,585)],arb,dram,both=True)
    d.label(42,603,'dashed = control',12)
    d.save()


def compiler():
    d=Diagram('compiler-flow',1040,540)
    d.group(250,40,510,180,'compile')
    model=d.node(30,105,170,70,'checkpoint','Llama / Qwen')
    front=d.node(280,105,180,70,'Python frontend','import + calibrate')
    comp=d.node(530,105,200,70,'C++ / MLIR','quantize, tile, schedule','blue')
    binary=d.node(810,105,200,70,'.llbin','instructions + weights')
    d.edge([(200,140),(280,140)],model,front)
    d.edge([(460,140),(530,140)],front,comp)
    d.edge([(730,140),(810,140)],comp,binary)
    ref=d.node(30,320,170,70,'Transformers','float reference','purple')
    golden=d.node(280,320,200,70,'NumPy','integer reference','purple')
    check=d.node(550,320,180,70,'compare outputs','integer agreement')
    run=d.node(810,320,200,70,'execute','ISA simulator / RTL','green')
    d.edge([(115,175),(115,320)],model,ref)
    d.edge([(630,175),(630,260),(380,260),(380,320)],comp,golden)
    d.label(387,248,'quantized graph')
    d.edge([(910,175),(910,320)],binary,run)
    d.edge([(480,355),(550,355)],golden,check)
    d.edge([(810,355),(730,355)],run,check)
    d.edge([(115,390),(115,470),(910,470),(910,390)],ref,run,both=True)
    d.label(398,457,'measure quantization error')
    d.save()


def generation():
    d=Diagram('token-generation',1080,680)
    d.group(30,35,1020,195,'prefill · process the prompt once')
    a=d.node(60,105,180,70,'prompt tokens','host: stage embeddings')
    b=d.node(310,105,190,70,'decoder layers','process prompt chunks','blue')
    c=d.node(570,105,180,70,'output projection','last prompt position','blue')
    e=d.node(820,105,200,70,'first new token','host: argmax')
    d.edge([(240,140),(310,140)],a,b)
    d.edge([(500,140),(570,140)],b,c)
    d.edge([(750,140),(820,140)],c,e)
    d.group(30,280,1020,280,'decode · repeat for each new token')
    f=d.node(60,350,180,70,'token embedding','host: stage one row')
    g=d.node(310,350,190,70,'decoder layers','process one position','blue')
    h=d.node(570,350,180,70,'output projection','next-token logits','blue')
    j=d.node(820,350,200,70,'next token','host: argmax')
    d.edge([(920,175),(920,250),(270,250),(270,330),(150,330),(150,350)],e,f)
    d.edge([(240,385),(310,385)],f,g)
    d.edge([(500,385),(570,385)],g,h)
    d.edge([(750,385),(820,385)],h,j)
    d.edge([(920,420),(920,510),(150,510),(150,420)],j,f)
    d.label(410,495,'continue until the stop condition')
    d.node(210,605,660,50,'per layer: Q/K/V → RoPE → write KV → attention → feed-forward',color='green')
    d.label(260,590,'KV stays in DRAM between steps; attention includes the current position')
    d.save()


def timeline():
    d=Diagram('execution-timeline',1080,580)
    d.label(40,40,'two weight buffers · schematic timing, not measured cycles',15)
    d.group(30,80,1020,400,'overlapped schedule')
    d.label(60,191,'DMA',16)
    d.label(60,301,'GEMM',16)
    d.label(60,411,'buffer reuse',14)
    d.edge([(220,115),(1000,115)])
    d.label(950,101,'time →')
    for x,title,detail in [(220,'load 0','buffer A'),(380,'load 1','buffer B'),(580,'load 2','buffer A'),(780,'load 3','buffer B')]:
        d.node(x,150,140,65,title,detail,'green')
    for x,title,detail in [(380,'compute 0','read A'),(580,'compute 1','read B'),(780,'compute 2','read A')]:
        d.node(x,260,180,65,title,detail,'blue')
    d.edge([(360,215),(360,245),(395,245),(395,260)],dashed=True)
    d.edge([(510,215),(510,240),(595,240),(595,260)],dashed=True)
    d.edge([(710,215),(710,240),(795,240),(795,260)],dashed=True)
    d.edge([(560,305),(570,305),(570,400),(610,400)],dashed=True)
    d.label(620,405,'A is free for load 2')
    d.label(220,445,'load 1 overlaps compute 0; load 2 overlaps compute 1')
    d.label(40,525,'wait for a load before reading its buffer; wait for its reader before overwriting it.')
    d.label(40,550,'semaphores enforce dependencies. Actual overlap depends on memory stalls and tile sizes.')
    d.save()


if __name__ == '__main__':
    architecture()
    compiler()
    generation()
    timeline()
