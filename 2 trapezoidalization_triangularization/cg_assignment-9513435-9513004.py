import matplotlib.pyplot as plt
import numpy as np

class Point:
    def __init__(self, coords):
        self.coords = coords
        self.edge = None
        self.ear = False
        self.next = None
        self.prev = None

    def getCoords(self):
        return self.coords
    
    def setIncidentEdge(self, edge):
        self.edge = edge
    def getIncidentEdge(self):
        return self.edge
    
    def getOutgoingEdges(self):
        visited = set()
        out = []
        here = self.edge
        while here and here not in visited:
            out.append(here)
            visited.add(here)
            temp = here.getTwin()
            if temp:
                here = temp.getNext()
            else:
                here = None
        return out

    
class Edge:
     def __init__(self):
        self.twin = None
        self.origin = None
        self.face = None
        self.next = None
        self.prev = None

     def setTwin(self, twin):
         self.twin = twin
     def getTwin(self):
         return self.twin
     def setOrigin(self, o):
         self.origin = o
     def getOrigin(self):
         return self.origin
     def setPrev(self, edge):
         self.edge = edge
     def getPrev(self):
         return self.prev
     def setNext(self, edge):
         self.next = edge
     def getNext(self):
         return self.next
     def setFace(self, face):
         self.face = face
     def getFace(self):
         return self.face

     def getFaceBoundary(self):
         visited = []
         boundary = []
         here = self
         while here and here not in visited:
             boundary.append(here)
             visited.append(here)
             here = here.getNext()
         return boundary

class Face:
    def __init__(self):
        self.outer = None
        self.inner = set()
        self.isolated = set()
    def getOuterComponents(self):
        return self.outer
    def setOuterComponents(self, edge):
        self.outer = edge
    def getOuterBoundary(self):
        if self.outer:
            return self.outer.getFaceBoundary()
        return []
    def getOuterBoundaryCoords(self):
        points = self.getOuterBoundary()
        points = [pt.origin.coords for pt in points]
        return points
    def getInnerComponents(self):
        return list(self.inner)
    def addInnerComponents(self, edge):
        self.inner.add(edge)
    def removeInnerComponents(self):
        self.inner.discard(edge)
    def getIsolatedVertex(self):
        return list(self.isolated)
    def addIsolatedVertex(self, point):
        self.isolated.add(point)
    def removeIsolatedVertex(self, point):
        self.isolated.discard(point)
    def getIsolatedVertices(self):
        return list(self.isolated)
        

class DCEL:
    def __init__(self):
        self.exterior = Face()
    def getExteriorFace(self):
        return self.exterior
    def getFaces(self):
        result = []
        known = set()
        known.add(self.exterior)
        temp = []
        temp.append(self.exterior)
        while temp:
            face = temp.pop(0)
            result.append(face)
            for edge in face.getOuterBoundary():
                nb=edge.getTwin().getFace()
                if nb and nb not in known:
                    temp.append(nb)
                    known.add(nb)
            for inner in face.getInnerComponents():
                for e in inner.getFaceBoundary():
                    nb = e.getTwin().getFace()
                    if nb and nb not in known:
                        temp.append(nb)
                        known.add(nb)
        return result

    def getEdges(self):
        edges = set()
        for f in self.getFaces():
            edges.update(f.getOuterBoundary())
            for inner in f.getInnerComponents():
                edges.update(f.getInnerComponents())
        return edges

    def getVertices(self):
        vertices = set()
        for f in self.getFaces():
            vertices.update(f.getIsolatedVertices())
            vertices.update([edge.getOrigin() for edge in f.getOuterBoundary()])
            for inner in f.getInnerComponents():
                vertices.update([edge.getOrigin() for edge in inner.getFaceBoundary()])
        return vertices        
                            
        
def buildPolygon(points):
    d = DCEL()
    exterior, interior = d.getExteriorFace(), Face()
    verts = []
    for p in points:
        verts.append(Point(p))
    innerEdges = []
    outerEdges = []
    for i in range(len(verts)):
        e = Edge()
        e.setOrigin(verts[i])
        verts[i].setIncidentEdge(e)
        e.setFace(interior)
        e2 = Edge()
        e2.setOrigin(verts[(i+1)%len(verts)])
        e2.setFace(exterior)
        e2.setTwin(e)
        e.setTwin(e2)
        innerEdges.append(e)
        outerEdges.append(e2)
        verts[i].next = verts[(i+1)%len(verts)]
    for i in range(len(verts)):
        innerEdges[i].setNext(innerEdges[(i+1)%len(verts)])
        innerEdges[i].setPrev(innerEdges[i-1])
        outerEdges[i].setNext(outerEdges[i-1])
        outerEdges[i].setPrev(outerEdges[(i+1)%len(verts)])
    interior.setOuterComponents(innerEdges[0])
    exterior.addInnerComponents(outerEdges[0])
    return d


class trapEdge(object):
    def __init__(self, a, b, p, le, re):
        self.left = a
        self.right = b
        self.pivot = p
        self.le = le # Left edge
        self.re = re # Right edge

class Point2(object):
    def __init__(self, a, b):
        self.x = a
        self.y = b
        
def orientation(p1, p2, p3):
    val = (p2.y - p1.y)*(p3.x - p2.x) - (p2.x - p1.x)*(p3.y-p2.y)
    if val == 0:
        return 0
    if val > 0:
        return 1
    return 2

def OnSegment(p1, p2, p3):
    if p2.x <= max(p1.x, p3.x) and p2.x >= min(p1.x, p3.x) and p2.y <= max(p1.y, p3.y) and p2.y >= min(p1.y, p3.y):
        return True
    return False

def find(A, B, C, D):
    a1, a2 = B.y - A.y, D.y - C.y
    b1, b2 = A.x - B.x, C.x - D.x
    c1, c2 = a1*(A.x) + b1*(A.y), a2*(C.x) + b2*(C.y)
    det = a1*b2 - a2*b1
    x = (b2*c1 - b1*c2)/det
    y = (a1*c2 - a2*c1)/det
    return (x,y)

def Intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    A, B, C, D = p1, q1, p2, q2
    a1, a2 = B.y - A.y, D.y - C.y
    b1, b2 = A.x - B.x, C.x - D.x
    c1, c2 = a1*(A.x) + b1*(A.y), a2*(C.x) + b2*(C.y)
    det = a1*b2 - a2*b1
    if det == 0:
        return False
    if o1!=o2 and o3!=o4:
        return True
    if o1 == 0 and OnSegment(p1, p2, q1):
        return True
    if o2 == 0 and OnSegment(p1, q2, q1):
        return True
    if o3 == 0 and OnSegment(p2, p1, q2):
        return True
    if o4 == 0 and OnSegment(p2, q1, q2):
        return True
    return False

def findIntersections(lines, hlines):
    res = {}
    for hline in hlines:
        p1 = Point2(hline[0], hline[1])
        q1 = Point2(hline[2], hline[3])
        for line in lines:
            p2 = Point2(line[0][0], line[0][1])
            q2 = Point2(line[0][2], line[0][3])
            if Intersect(p1, q1, p2, q2):
                res[find(p1, q1, p2, q2)] = line[1]
    return res
                                                                

def findTrapEdges(d):
    N = len(d.getVertices())
    verts = [list(d.getVertices())[i].getCoords() for i in range(N)]
    verts = list(zip(verts, [v for v in list(d.getVertices())]))
    edges = [(verts[i][1].coords, verts[i][1].next.coords) for i in range(N)]
    edges = list(zip(edges, [v[1].getOutgoingEdges()[0] for v in verts]))
    verts.sort(key=lambda v: -v[0][0])
    lines = []
    temp = []
    for e in edges:
        temp = [e[0][0][0], e[0][0][1], e[0][1][0], e[0][1][1]], e[1]
        lines.append(temp)
    lines2 = []
    temp = []
    for v in verts:
        temp = verts[0][0][0], v[0][1], verts[-1][0][0], v[0][1]
        lines2.append(temp)

    res = findIntersections(lines, lines2)
    res = [[x, y, res[(x,y)]] for (x,y) in res]
    res.sort(key=lambda x: -x[1])

    ret = []
    for v in verts:
        templ = [(x[0],x[1],x[2]) for x in res if x[0] < v[0][0] and x[1] == v[0][1]]
        tempr = [(x[0],x[1],x[2]) for x in res if x[0] > v[0][0] and x[1] == v[0][1]]
        templ.sort(key = lambda x: x[0])
        tempr.sort(key = lambda x: x[0])
        if (len(templ)%2 == 0 and len(tempr)%2 == 0):
            if v[1].getOutgoingEdges()[0].getTwin().origin.getCoords()[1] < v[1].getCoords()[1]:
                tr = trapEdge(v[0], v[0], v[1], v[1].getOutgoingEdges()[0], v[1].getOutgoingEdges()[1].getTwin())
            else:
                tr = trapEdge(v[0], v[0], v[1], v[1].getOutgoingEdges()[1], v[1].getOutgoingEdges()[0])
            ret.append(tr)

        if (len(templ)%2 == 1 and len(tempr)%2 == 1):
            tr = trapEdge(templ[-1][:2], tempr[0][:2], v[1], templ[-1][2], tempr[0][2])
            ret.append(tr)
            
        if (len(templ)%2 == 0 and len(tempr)%2 == 1):
            tr = trapEdge(v[0], tempr[0][:2], v[1], v[1].getOutgoingEdges()[0], tempr[0][2])
            ret.append(tr)

        if (len(templ)%2 == 1 and len(tempr)%2 == 0):
            tr = trapEdge(templ[-1][:2], v[0], v[1], templ[-1][2], v[1].getOutgoingEdges()[1].getTwin())
            ret.append(tr)

    return ret        
        

def getMonotonePartitioningDiagonals(d):
    diagonals=[]
    ret = findTrapEdges(d)
    ret = sorted(ret, key = lambda x:-x.pivot.coords[1])
    a=dict()
    b=dict()
    for x in ret:
        x.le = x.re.getTwin()
        if x.pivot.coords[1] > x.re.getTwin().origin.coords[1]:
            a[x.pivot] = (x.le, x.re)
            if x.le in b:
                b[x.le].append(x.pivot)
            else:
                b[x.le] = [x.pivot]
            if x.re in b:
                b[x.re].append(x.pivot)
            else:
                b[x.re] = [x.pivot]
    for edge in b:
        b[edge].append(edge.getTwin().origin)

    for pt in sorted(a, key=lambda x:-x.coords[1]):
        if pt in (a[pt][0].origin, a[pt][0].getTwin().origin):
            diagonals.append((pt, b[a[pt][1]][b[a[pt][1]].index(pt)+1]))
        elif pt in (a[pt][1].origin, a[pt][1].getTwin().origin):
            diagonals.append((pt, b[a[pt][0]][b[a[pt][0]].index(pt)+1]))
        else:
            diagonals.append((pt,
                              min(b[a[pt][0]][b[a[pt][0]].index(pt)+1], b[a[pt][1]][b[a[pt][1]].index(pt)+1],
                                  key=lambda p:p.coords[1])))
    diagonals = list(set(diagonals)-set([(x.origin, x.getTwin().origin) for x in d.getEdges()]))
    diagonals = list(set(diagonals)-set([(x.getTwin().origin, x.origin) for x in d.getEdges()]))
    return diagonals    


def insertDiag(d, p1, p2):
    points1 = []
    points2 = []
    d_points = d.getFaces()[1].getOuterBoundaryCoords()
    p1 = p1.coords
    p2 = p2.coords
    if (p1 in d_points and p2 in d_points) and p1!=p2:
        temp1 = min(d_points.index(p1), d_points.index(p2))
        temp2 = max(d_points.index(p1), d_points.index(p2))
        points1 = d_points[temp1:temp2+1]
        points2 = d_points[temp2:] + d_points[:temp1+1]
        d1 = buildPolygon(points1)
        d2 = buildPolygon(points2)
        return [d1, d2]    
    return [d]

def getAllMonotonePolygons(d, diagonals):
    allPolygonCoords = []
    allPolygons = [d]
    while diagonals != []:
        next_d = diagonals.pop(0)
        allPolygons = [insertDiag(p, next_d[0], next_d[1]) for p in allPolygons]
        allPolygons = [n for nl in allPolygons for n in nl]        
               
    for polygon in allPolygons:
        L = []
        for point in polygon.getFaces()[1].getOuterBoundaryCoords():
            L.append((point[0], point[1]))
        L.append(L[0])    
        allPolygonCoords.append(L)
    return allPolygonCoords    




def angleCal(vector1,vector2):
    cosTh = np.dot(vector1,vector2)
    sinTh = np.cross(vector1,vector2)
    theta = np.rad2deg(np.arctan2(sinTh,cosTh))
    return theta

def isConvex(a,b,c,dirc):
    # we determine whether "c" can see "a" or not 
    # is it can see return 1 O.W return 0
    t=0 # final theta
    ab = [(a[0]-b[0]),(a[1]-b[1])] # ab vector
    cb = [(c[0]-b[0]),(c[1]-b[1])] # cb vector
    base = [1,0] # base vector
    if(dirc==0):
        t = angleCal(base,ab)+ abs(angleCal(base,cb))
    else:
        t = (180-angleCal(base,ab))+(180+angleCal(base,cb)) 
        
    if(t>180):
        return 0
    return 1


def extraFeatureAdder(list):
    listExt=[] # list with extra features 
    lower=[-1,1000] # [lower point index in list , y-value]
    upper=[-1,-1000] # [upper point index in list , y-value]
    repeatedPoint = list.pop()
    i=0
    for point in list:
        if(point[1]>upper[1]):
            upper=[i,point[1]]
        if(point[1]<lower[1]):
            lower=[i,point[1]]
        i+=1
    
    smallerIndex=0
    greaterIndex=0
    if(lower[0]<upper[0]):
        smallerIndex= lower[0]
        greaterIndex= upper[0]
    else:
        smallerIndex= upper[0]
        greaterIndex= lower[0]
        
    dirc = 0 #direction
    
    if(list[smallerIndex][0] < list[(smallerIndex+1)][0]):
        dirc = 1
    elif(list[smallerIndex][0] == list[(smallerIndex+1)][0]):
        before = smallerIndex -1
        if(before<0):
            before = len(list)-1
        if(list[before][0]<list[smallerIndex][0]):
            dirc=1 
        
    k=0 # I add this to neutralize side efect of poping func **
    for i in range(greaterIndex-smallerIndex):
        listExt.append((list.pop(smallerIndex+i-k),dirc)) # **
        k+=1     
    #changing direction 
    if(dirc==1):
        dirc=0
    else:
        dirc=1       
    # adding the rest of points in listExt
    for i in range(len(list)):
        listExt.append((list[i],dirc))
        
    # sorting listExt 
    listExt.sort(key=lambda x: x[0][1], reverse=False)
    # Since Polygons are monotonic we cas also sort in linear time
    #print("PolyExt:",listExt)    
    return listExt
        


# Triangularization 
# we receive n number of monotonic polygon and then do the job(Trianglu...)
# Format : CloseListOfPolygonVertices = [A,B,...,F,A] , A=[A_x, A_y]


def triangularization(list , n):
    diagonal=[]
    # List of diagonals # at the end we draw this 
    #format of diagonal list : [ ( startOfDiagonal , endOfDiagonal ) , ... ] & start/endOfDiagonal = [x,y]
    # Finding all diagonals 
    for i in range(0,n):
        poly = list[i]
        polyExt = extraFeatureAdder(poly)#listOf poly ver + extra features
        
        s=[polyExt.pop(),polyExt.pop()] # stack containig first two vertecis
        #print("stack:",s)
        #print("polyExt before intering:",polyExt)
        for j in range((len(polyExt)-1),-1,-1): # iterating all except last one(lower point)
            # Reversed for loop (from the top of list to down)
            if( s[(len(s)-1)][1] != polyExt[j][1]   ):
                topStack = s.pop()
                diagonal.append( ( polyExt[j][0] , topStack[0] ))
                s.pop(0) # get rid of the first element in stack
                for i in range(len(s)):
                    diagonal.append( ( polyExt[j][0] , s[i][0] ))
                s = [] # clearing the stack
                s = [topStack,polyExt[j]]
            else:
                lastPoped = s.pop()
                while(1):
                    if(len(s)>0):
                        if( isConvex(s[(len(s)-1)][0],lastPoped[0],polyExt[j][0],polyExt[j][1]) == 1 ):
                            diagonal.append( (  polyExt[j][0] ,s[(len(s)-1)][0]  ) )
                            lastPoped= s.pop()
                        else:
                            break
                    else:
                        break
                        
                s.append(lastPoped)
                s.append(polyExt[j])
            
        # at the end of last for loop:
        # add diagonal from polyExt[(len(polyExt)-1)][0] to all remaining vertices in s except the first and last one
        if(len(s)>2):
            s.pop() #poping the last one
            s.pop(0) #poping the first one
            if(len(s)>0):
                for pointExt in s:
                    diagonal.append( (  polyExt[(len(polyExt)-1)][0] ,pointExt[0]  ) )
    
    return diagonal

def draw(d, ret, p_diags, tri_diags):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    e = list(d.getEdges())[0]
    visited = []
    for d1 in tri_diags:
        ax1.plot([d1[0][0],d1[1][0]], [d1[0][1], d1[1][1]], color="k")
        
    for d in p_diags:
        ax1.plot([d[0].coords[0], d[1].coords[0]],[d[0].coords[1], d[1].coords[1]], color="green")    

    while True:
        ax1.plot([e.getOrigin().getCoords()[0], e.getNext().getOrigin().getCoords()[0]],
                 [e.getOrigin().getCoords()[1], e.getNext().getOrigin().getCoords()[1]], color="blue")
        if e in visited:
            break
        visited.append(e)
        e = e.getNext()

    for d in p_diags:
        ax1.plot([d[0].coords[0], d[1].coords[0]],[d[0].coords[1], d[1].coords[1]], color="green")
        
    for r in ret:
        ax1.plot([r.left[0],r.right[0]], [r.left[1], r.right[1]], color="red", marker=".")    

    plt.show()

    
points = []
print("Example: if you want a polygon with these points (4,4.5), (4,2), (3,1), (1.5,1.5), (1,4), (3,3) as Vertices")
print("Enter points in this format:\n4 4.5\n4 2\n3 1\n1.5 1.5\n1 4\n3 3")
print("Enter points:(Press Enter at the end)")
while True:
    p = list(map(float, input('').split()))
    if not p:
        break
    points.append((p[0],p[1]))

d = buildPolygon(points)
ret = findTrapEdges(d)
partitioning_diags = getMonotonePartitioningDiagonals(d)
allPolygons = getAllMonotonePolygons(d, partitioning_diags.copy())
allPolygons = [p for p in allPolygons if len(p)>3]
tri_diags = triangularization(allPolygons, len(allPolygons))
draw(d, ret, partitioning_diags, tri_diags)     
        
        
         
        
        
        
