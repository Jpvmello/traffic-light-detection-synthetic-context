import java.util.ArrayList;

class TrafficLight {
  float tlWidth, tlHeight;
  float xFromPostBase, yFromPostBase, zFromPostBase;
  boolean toMark;

  public TrafficLight(float x, float y, float z, boolean toMark) {
    this.tlWidth = random(80, 120);
    this.tlHeight = random(280, 320);

    this.xFromPostBase = x;
    this.yFromPostBase = y;
    this.zFromPostBase = z;
    this.toMark = toMark;

    if (toMark)
      tls++;
  }
  
  public boolean isToMark() {
    return toMark;
  }
  
  public void draw() {
    pg.pushMatrix();
      pg.translate(xFromPostBase, yFromPostBase, zFromPostBase);
      pg.rotateX(random(0.45*PI, 0.55*PI));
      pg.rotateZ(random(-0.02*PI, 0.02*PI));
      pg.fill(50);     
      
      pg.translate(0, 0, -0.1*tlHeight);
      
      if (context || toMark) pg.box(tlWidth, tlHeight, 0.2*tlHeight);
      pg.translate(0, 0, -0.1*tlHeight);
      
      float screenv1x = pg.screenX(-0.5*tlWidth, -0.5*tlHeight, 0);
      float screenv1y = pg.screenY(-0.5*tlWidth, -0.5*tlHeight, 0);
      float screenv2x = pg.screenX(+0.5*tlWidth, -0.5*tlHeight, 0);
      float screenv2y = pg.screenY(+0.5*tlWidth, -0.5*tlHeight, 0);
      float screenv3x = pg.screenX(+0.5*tlWidth, +0.5*tlHeight, 0);
      float screenv3y = pg.screenY(+0.5*tlWidth, +0.5*tlHeight, 0);
      float screenv4x = pg.screenX(-0.5*tlWidth, +0.5*tlHeight, 0);
      float screenv4y = pg.screenY(-0.5*tlWidth, +0.5*tlHeight, 0);

      float screenX1 = min(new float [] {screenv1x, screenv2x, screenv3x, screenv4x});
      float screenY1 = min(new float [] {screenv1y, screenv2y, screenv3y, screenv4y});
      float screenX2 = max(new float [] {screenv1x, screenv2x, screenv3x, screenv4x});
      float screenY2 = max(new float [] {screenv1y, screenv2y, screenv3y, screenv4y});

      BoundingBox boundingBox = new BoundingBox(screenX1, screenY1, screenX2, screenY2);
      if (screenX2 >= 0 && screenX1 < width && screenY2 >= 0 && screenY1 < height && toMark) {
          if (!isOccluded(boundingBox, screenX1, screenY1, screenX2, screenY2)) {
            boundingBoxes.add(boundingBox);
          }
          else
            toMark = false;
      }
      
      color yellow = color(255, random(100, 180), random(0, 50));
      color red = color(255, random(0, 100), random(0, 50));
      color green = color(random(0, 50), 255, random(50, 180));
      
      pg.pushMatrix();
        pg.pushMatrix();
          pg.rotateX(PI);
          pg.fill(50);
          if (context || toMark) cylinderPart(0.1*tlHeight + 2, 0.2*tlHeight, 180, 300);
        pg.popMatrix();
        if (boundingBox.getCls() == "yellow") {
          //setLightsOn(false);
          //pg.pointLight(255, 255, 0, 0, 0, 0);
          pg.emissive(yellow);
        } else {
          if (random(1) < 0.3) { //Counter
            color center = 0;
            if (boundingBox.getCls() == "red") center = red;
            if (boundingBox.getCls() == "green") center = green;
            for (int scale = -1; scale <= 1; scale += 2) {
              pg.pushMatrix();
                pg.translate(0, 0, -0.05*tlHeight);
                pg.scale(scale, 1, 1);
                pg.emissive(0);
                pg.pushMatrix();
                  pg.translate(-0.05*tlHeight, 0, 0);
                  if (random(1) < 0.5)
                    pg.emissive(center);
                  if (context || toMark) pg.box(0.02*tlHeight, 0.1*tlHeight, 1);
                pg.popMatrix();
                pg.emissive(0);
                pg.pushMatrix();
                  pg.translate(-0.01*tlHeight, 0, 0);
                  if (random(1) < 0.5)
                    pg.emissive(center);
                  if (context || toMark) pg.box(0.02*tlHeight, 0.1*tlHeight, 1);
                pg.popMatrix();
                pg.emissive(0);
                pg.pushMatrix();
                  pg.translate(-0.03*tlHeight, -0.05*tlHeight, 0);
                  if (random(1) < 0.5)
                    pg.emissive(center);
                  if (context || toMark) pg.box(0.04*tlHeight, 0.02*tlHeight, 1);
                pg.popMatrix();
                pg.emissive(0);
                pg.pushMatrix();
                  pg.translate(-0.03*tlHeight, 0.05*tlHeight, 0);
                  if (random(1) < 0.5)
                    pg.emissive(center);
                  if (context || toMark) pg.box(0.04*tlHeight, 0.02*tlHeight, 1);
                pg.popMatrix();
                pg.emissive(0);
                pg.pushMatrix();
                  pg.translate(-0.03*tlHeight, 0, 0);
                  if (random(1) < 0.5)
                    pg.emissive(center);
                  if (context || toMark) pg.box(0.04*tlHeight, 0.02*tlHeight, 1);
                pg.popMatrix();
              pg.popMatrix();
            }
          }
          
          pg.emissive(0);
          pg.fill(100, 100);
        }
        if (context || toMark) pg.sphere(0.1*tlHeight);
        pg.emissive(0);//setLightsOn(true);
      pg.popMatrix();
      pg.pushMatrix();
        pg.translate(0, tlHeight/4, 0);
        pg.pushMatrix();
          pg.rotateX(PI);
          pg.fill(50);
          if (context || toMark) cylinderPart(0.1*tlHeight + 2, 0.2*tlHeight, 180, 300);
        pg.popMatrix();
        boolean arrow = false;
        if (boundingBox.getCls() == "red") {
          //setLightsOn(false);
          //pg.pointLight(255, 0, 0, 0, 0, 0);
          arrow = random(1) < 0.3;
          if (arrow) {
            pg.pushMatrix();
              if (random(1) < 0.5)
                pg.scale(-1, 1, 1);
              pg.emissive(red);
              pg.translate(0, 0, -0.05*tlHeight);
              //pg.scale(red, 1, 1);
              pg.pushMatrix();
                if (context || toMark) pg.box(0.12*tlHeight, 0.04*tlHeight, 1);
              pg.popMatrix();
              pg.pushMatrix();
                pg.translate(-0.06*tlHeight, -0.03*tlHeight, 0);
                pg.rotateZ(-PI/4);
                if (context || toMark) pg.box(0.1*tlHeight, 0.04*tlHeight, 1);
              pg.popMatrix();
              pg.pushMatrix();
                pg.translate(-0.06*tlHeight, 0.03*tlHeight, 0);
                pg.rotateZ(PI/4);
                if (context || toMark) pg.box(0.1*tlHeight, 0.04*tlHeight, 1);
              pg.popMatrix();
            pg.popMatrix();
            
            pg.emissive(0);
            pg.fill(100, 100);
          } else
            pg.emissive(red);
        } else pg.fill(100, 100);
        if (context || toMark) pg.sphere(0.1*tlHeight);
        pg.emissive(0);//setLightsOn(true);
      pg.popMatrix();
      pg.pushMatrix();
        pg.translate(0, -tlHeight/4, 0);
        pg.pushMatrix();
          pg.rotateX(PI);
          pg.fill(50);
          if (context || toMark) cylinderPart(0.1*tlHeight + 2, 0.2*tlHeight, 180, 300);
        pg.popMatrix();
        if (boundingBox.getCls() == "green") {
          //setLightsOn(false);
          //pg.pointLight(255, 0, 0, 0, 0, 0);
          arrow = random(1) < 0.3;
          if (arrow) {
            pg.pushMatrix();
              if (random(1) < 0.5)
                pg.scale(-1, 1, 1);
              pg.emissive(green);
              pg.translate(0, 0, -0.05*tlHeight);
              //pg.scale(red, 1, 1);
              pg.pushMatrix();
                if (context || toMark) pg.box(0.12*tlHeight, 0.04*tlHeight, 1);
              pg.popMatrix();
              pg.pushMatrix();
                pg.translate(-0.06*tlHeight, -0.03*tlHeight, 0);
                pg.rotateZ(-PI/4);
                if (context || toMark) pg.box(0.1*tlHeight, 0.04*tlHeight, 1);
              pg.popMatrix();
              pg.pushMatrix();
                pg.translate(-0.06*tlHeight, 0.03*tlHeight, 0);
                pg.rotateZ(PI/4);
                if (context || toMark) pg.box(0.1*tlHeight, 0.04*tlHeight, 1);
              pg.popMatrix();
            pg.popMatrix();
            
            pg.emissive(0);
            pg.fill(100, 100);
          } else
            pg.emissive(green);
        } else pg.fill(100, 100);
        if (context || toMark) pg.sphere(0.1*tlHeight);
        pg.emissive(0);//setLightsOn(true);
      pg.popMatrix();
    pg.popMatrix(); 
  } 
}

class Post {
  int postColor;
  float radius, pheight, length;
  float distCenterX, distCenterY;
  boolean frontal, oppositeSide;
  TrafficLight postTrafficLight;
  TrafficLight lengthTrafficLight1;
  TrafficLight lengthTrafficLight2;

  public Post(float radius, float pheight, float length, boolean frontal, float distCenterX, float distCenterY) {
    this.postColor = int(random(200));
    this.radius = radius;
    this.pheight = pheight;
    this.length = 0;
    this.frontal = frontal;
    this.oppositeSide = random(1) < 0.4;
    this.distCenterX = distCenterX;
    this.distCenterY = distCenterY;
    this.postTrafficLight = null;
    this.lengthTrafficLight1 = null;
    this.lengthTrafficLight2 = null;

    boolean hasHorizontal = random(1) < 0.9;
    if (hasHorizontal) {
      this.length = length;
      this.lengthTrafficLight1 = new TrafficLight(-random(0.8, 1)*length, 0.5*radius + 30, 0.9*pheight, frontal);
      if (random(1) < 0.3)
        this.lengthTrafficLight2 = new TrafficLight(-random(0.4, 0.6)*length, 0.5*radius + 30, 0.9*pheight, frontal);
    }
    
    if (!hasHorizontal || random(1) < 0.9)
      this.postTrafficLight = new TrafficLight(0, radius + 30, random(0.4, 0.6)*pheight, frontal);
  }

  public void draw() {
    pg.pushMatrix();
      if (oppositeSide) {
        pg.scale(-1, 1, 1);
        //pg.rotateZ(PI);
      }
      pg.translate(distCenterX, distCenterY);
      //if (oppositeSide)
        //pg.rotateZ(PI);
      pg.fill(postColor);
      if (context) cylinder(radius, pheight);
      pg.pushMatrix();
        pg.translate(0, 0.5*radius, 0.9*pheight);
        pg.rotateY(-PI/2);
        if (context) cylinder(0.5*radius, length);
      pg.popMatrix();
      if (postTrafficLight != null) {
        postTrafficLight.draw();
      }
      if (length > 0) {
        if (lengthTrafficLight1 != null)
          lengthTrafficLight1.draw();
        if (lengthTrafficLight2 != null)
          lengthTrafficLight2.draw();
      }
    pg.popMatrix();
  }
}

class RoadPart {
  int pwidth, length;
  int roadColor;
  int lanes;
  boolean front, crossWalk, continuousDiv, reservation;
  Post post;
  ArrayList<Integer> carsByLane;

  public RoadPart(int roadColor, boolean hasPost, boolean frontOfTrafficLight) {
    this.roadColor = roadColor;
    this.lanes = 2 * int(random(1, 4));
    this.front = frontOfTrafficLight;
    this.pwidth = lanes * width;
    this.length = 20*height;
    this.crossWalk = random(1) < 0.7;
    this.continuousDiv = lanes > 2 || (random(1) < 0.5);
    this.reservation = false;//lanes == 8 || (lanes > 2  && random(1) < 0.5);
    this.post = null;

    this.carsByLane = new ArrayList();
    for (int i = 0; i < lanes; i++)
      this.carsByLane.add(new Integer(int(random(6))));
    
    if (hasPost)
      post = new Post(random(0.01, 0.015) * this.pwidth, random(1.5, 2)*width, this.pwidth/2, frontOfTrafficLight, 0.6*this.pwidth, 0.05*this.length);
  }

  public int getWidth() {
    return this.pwidth;
  }

  public int getLength() {
    return length;
  }

  public float randomRightLane() {
    return int(random(1, lanes/2 + 1)) * width;
  }

  public void draw() {
    pg.pushMatrix();
      if (post != null) {
        pg.pushMatrix();
            pg.translate(0, 3 * this.pwidth/lanes, 1.5);
            post.draw();
        pg.popMatrix();
      }
    
      pg.noStroke();
      pg.fill(roadColor);
      if (context) pg.rect(0, length/2, this.pwidth, this.length);

      int nlane = 0;
      for (float x = 0.5*this.pwidth/lanes - this.pwidth/2; x < this.pwidth/2; x += this.pwidth/lanes) {
        float distCars = this.length/(carsByLane.get(nlane) + 1);
        for (float y = distCars; y < this.length; y += distCars) {
          if (!this.front || width/2 + x != lane || height/2 + y < distance - 0.5*distCars || height/2 + y > distance + 0.5*distCars) {
             pg.pushMatrix();
              pg.translate(x, y, 0);
              pg.rotateX(PI/2);
              if (x < 0)
                pg.rotateY(-PI/2);
              else
                pg.rotateY(PI/2);
              pg.scale(5);
              //pg.scale(10);

              if (this.front && height/2 + y < distance) {
                float[] v1 = new float[] {-0.5*carW, +1*carH,  -0.5*carD};
                float[] v2 = new float[] {+0.5*carW, +1*carH,  -0.5*carD};
                float[] v3 = new float[] {+0.5*carW, +1*carH,  +0.5*carD};
                float[] v4 = new float[] {-0.5*carW, +1*carH,  +0.5*carD};
                float[] v5 = new float[] {-0.5*carW, +0.0*carH, +0.5*carD};
                float[] v6 = new float[] {+0.5*carW, +0.0*carH, +0.5*carD};
                float[] v7 = new float[] {+0.5*carW, +0.0*carH, -0.5*carD};
                float[] v8 = new float[] {-0.5*carW, +0.0*carH, -0.5*carD};
                
                float screenv1x = pg.screenX(v1[0], v1[1], v1[2]);
                float screenv2x = pg.screenX(v2[0], v2[1], v2[2]);
                float screenv3x = pg.screenX(v3[0], v3[1], v3[2]);
                float screenv4x = pg.screenX(v4[0], v4[1], v4[2]);
                float screenv5x = pg.screenX(v5[0], v5[1], v5[2]);
                float screenv6x = pg.screenX(v6[0], v6[1], v6[2]);
                float screenv7x = pg.screenX(v7[0], v7[1], v7[2]);
                float screenv8x = pg.screenX(v8[0], v8[1], v8[2]);
                float screenv1y = pg.screenY(v1[0], v1[1], v1[2]);
                float screenv2y = pg.screenY(v2[0], v2[1], v2[2]);
                float screenv3y = pg.screenY(v3[0], v3[1], v3[2]);
                float screenv4y = pg.screenY(v4[0], v4[1], v4[2]);
                float screenv5y = pg.screenY(v5[0], v5[1], v5[2]);
                float screenv6y = pg.screenY(v6[0], v6[1], v6[2]);
                float screenv7y = pg.screenY(v7[0], v7[1], v7[2]);
                float screenv8y = pg.screenY(v8[0], v8[1], v8[2]);
                
                float screenX1 = min(new float [] {screenv1x, screenv2x, screenv3x, screenv4x,
                    screenv5x, screenv6x, screenv7x, screenv8x});
                float screenY1 = min(new float [] {screenv1y, screenv2y, screenv3y, screenv4y,
                    screenv5y, screenv6y, screenv7y, screenv8y});
                float screenX2 = max(new float [] {screenv1x, screenv2x, screenv3x, screenv4x,
                    screenv5x, screenv6x, screenv7x, screenv8x});
                float screenY2 = max(new float [] {screenv1y, screenv2y, screenv3y, screenv4y,
                    screenv5y, screenv6y, screenv7y, screenv8y});
                
                //if (screenX2 > 0 && screenX1 < width && screenY1 > 0 && screenY2 < height)
                if (occludes(screenX1, screenY1, screenX2, screenY2)) {
                    pg.popMatrix();
                    continue;
                }
              }

              color wheel = color(0);
              color hubcap = color(random(150, 200));
              color characters = color(0);
              color plate = color(random(150, 200));
              color bodywork = color(random(250), random(250), random(250));
              color details = color(0);
              color exhaust = color(random(50));
              color glass = color(random(100), 220);
              color glassContour = color(random(50));
              color[] colors = {
                wheel,          //0
                hubcap,         //1
                wheel,          //2
                hubcap,         //3
                wheel,          //4
                hubcap,         //5
                wheel,          //6
                hubcap,         //7
                characters,     //8
                characters,     //9
                plate,          //10
                bodywork,       //11
                bodywork,       //12
                bodywork,       //13
                glass,          //14
                characters,     //15
                characters,     //16
                characters,     //17
                plate,          //18
                bodywork,       //19
                exhaust,        //20
                details,        //21
                bodywork,       //22
                bodywork,       //23
                glass,          //24
                glass,          //25
                bodywork,       //26
                bodywork,       //27
                glassContour    //28
             };

              color cur = car.getChild(0).getFill(0);
              int clr = 0;
              for (int i = 0; i < car.getChildCount(); i++) {
                if (blackCarChildren.contains(i))
                    continue;
                PShape c = car.getChild(i);
                PShape cCopy = carCopy.getChild(i);
                if (c.getFill(0) != cur) {
                  cur = c.getFill(0);
                  clr++;
                }
                cCopy.setFill(colors[clr]);
              }
              float headlightBlue = random(0, 255);
              for (int multiplier = -1, headlight = 0; headlight < 2; headlight++, multiplier *= -1) {
                pg.pushMatrix();
                  pg.translate(85, 42.5, multiplier * 28);
                  pg.emissive(255, 255, headlightBlue);
                  float lightWidth = random(4, 5);
                  if (context) pg.box(lightWidth);
                  pg.emissive(0);
                  pg.translate(0, -2.5, 0);
                  pg.fill(200, 0, 0, 150);
                  if (context) pg.box(5, 10, 5);
                pg.popMatrix();
              }
              pg.pushMatrix();
                pg.translate(0, 0.5, 0);
                pg.rotateX(PI/2);
                pg.fill(0, 0, 0, 200);
                if (context) pg.ellipse(0, 0, 200, 100);
              pg.popMatrix();
              if (context) pg.shape(carCopy, 0, 0);
             pg.popMatrix();
          }
        }
        nlane++;
      }
      
      pg.pushMatrix();
          pg.translate(0, 3 * this.pwidth/lanes, 0.5);
          pg.fill(255);
          if (context)
            for (float y = 0.15*this.length; y <= this.length; y += 0.1*this.length)
              for (float x = this.pwidth/lanes - this.pwidth/2; x < this.pwidth/2; x += this.pwidth/lanes)
                pg.rect(x, y, this.pwidth/50, 0.05*this.length);

          pg.translate(0, 0, 0.5);
          pg.fill(255, 255, 0);
          if (context) 
            if (continuousDiv)
              pg.rect(0, 0.5*length, this.pwidth/20, 0.8*this.length);
            else
              for (float y = 0.15*this.length; y <= this.length; y += 0.1*this.length)
                pg.rect(0, y, this.pwidth/50, 0.05*this.length);

          pg.translate(0, 0, 0.5);
          pg.fill(255);
          if (context) pg.rect(0, 0.1*length, this.pwidth, 0.01*this.length);

          if (context) 
            if (crossWalk)
              for (float x = this.pwidth/10 - this.pwidth/2; x < this.pwidth/2; x += this.pwidth/10)
                pg.rect(x, 0.05*this.length, this.pwidth/20, 0.05*this.length);
      pg.popMatrix();
    pg.popMatrix();
  }
}

class Road {
  int roadColor;
  private ArrayList<RoadPart> roadParts;
  boolean front, left, right;

  public Road() {
    this.roadColor = 75;
    this.roadParts = new ArrayList();
    this.roadParts.add(new RoadPart(this.roadColor, true, true));

    this.front = random(1) < 0.8;
    this.left = random(1) < 0.8;
    this.right = (!front && !left) || (random(1) < 0.8);

    if (this.front)
      this.roadParts.add(new RoadPart(this.roadColor, random(1) < 0.7, false));
    else
      this.roadParts.add(null);
    if (this.left)
      this.roadParts.add(new RoadPart(this.roadColor, random(1) < 0.7, false));
    else
      this.roadParts.add(null);
    if (this.right)
      this.roadParts.add(new RoadPart(this.roadColor, random(1) < 0.7, false));
    else
      this.roadParts.add(null);
  }

  public float randomRightLane() {
    return roadParts.get(0).randomRightLane();
  }

  public void draw() {
    pg.pushMatrix();
      pg.rotateZ(PI);
      if (front)
        roadParts.get(1).draw();
      pg.rotateZ(PI/2);
      if (left)
        roadParts.get(2).draw();
      pg.rotateZ(PI);
      if (right)
        roadParts.get(3).draw();
      pg.rotateZ(-PI/2);
      roadParts.get(0).draw();
    pg.popMatrix();
  }
}

boolean occludes(float sx1, float sy1, float sx2, float sy2) {
    for (BoundingBox boundingBox: boundingBoxes) {
        boolean cond1 = (boundingBox.getX2() >= sx1 || boundingBox.getX1() <= sx2) &&
            (boundingBox.getY2() >= sy1 && boundingBox.getY1() <= sy2);
        boolean cond2 = (boundingBox.getY2() >= sy1 || boundingBox.getY1() <= sy2) &&
            (boundingBox.getX2() >= sx1 && boundingBox.getX1() <= sx2);
        if (cond1 || cond2) {
            float overlap_area = (min(boundingBox.getX2(), sx2) - max(boundingBox.getX1(), sx1)) *
                (min(boundingBox.getY2(), sy2) - max(boundingBox.getY1(), sy1));
            float occlusion = overlap_area/boundingBox.getArea();
            if (occlusion >= occlusionThresh)
                return true;
        }
    }
    return false;
}

boolean isOccluded(BoundingBox boundingBox, float sx1, float sy1, float sx2, float sy2) {
    float limitedx1 = max(0, sx1);
    float limitedy1 = max(0, sy1);
    float limitedx2 = min(width - 1, sx2);
    float limitedy2 = min(height - 1, sy2);
    float occlusion = 1 - ((limitedx2-limitedx1) * (limitedy2-limitedy1))/((sx2-sx1) * (sy2-sy1));
    if (occlusion >= occlusionThresh)
        return true;
    return false;
}

void cylinder(float radius, float cheight) {
  cylinderPart(radius, cheight, 0, 360);
}

void cylinderPart(float radius, float cheight, int initAng, int finalAng) {
  /*pg.beginShape();
    for (int ang = 0; ang < 360; ang++)
      pg.vertex(radius * cos(ang), radius * sin(ang), 0);
  pg.endShape();
  pg.beginShape();
    for (int ang = 0; ang < 360; ang++)
      pg.vertex(radius * cos(ang), radius * sin(ang), cheight);
  pg.endShape();*/
  for (int ang = initAng; ang < finalAng; ang++) {
    float rad = radians(ang);
    pg.beginShape();
      pg.vertex(radius * cos(rad), radius * sin(rad), 0);
      pg.vertex(radius * cos(rad + 1), radius * sin(rad + 1), 0);
      pg.vertex(radius * cos(rad + 1), radius * sin(rad + 1), cheight);
      pg.vertex(radius * cos(rad), radius * sin(rad), cheight);
    pg.endShape();
  }
}

void setLightsOn(boolean on) {
  lightsOn = on;
  if (on) {
    pg.directionalLight(255, 255, 255, lightX, 1, lightZ);
    pg.ambientLight(ambientLightComponent, ambientLightComponent, ambientLightComponent);
  } else
    pg.noLights();
}

String time() {
  int total_seconds = (millis() - startTime)/1000; 
  int hours = int(total_seconds/3600.0);
  int minutes = int((total_seconds/3600.0 - hours) * 60);
  int seconds = total_seconds - 3600*hours - 60*minutes;
  return String.valueOf(hours) + "h" + String.valueOf(minutes) + "m" + String.valueOf(seconds) + "s";
}

boolean isNumber(String entry) {
    char[] charArray = entry.toCharArray();
    for (char c: charArray) {
        if (c != '0' && c != '1' && c != '2'
            && c != '3' && c != '4' && c != '5' && c != '6' && c != '7' && c != '8' && c != '9')
            return false;
    }
    return true;
}

Road road;
PShape car, carCopy;
PGraphics pg;
int nImage = 1, nImages = 0, seed = 1, initialSeed;
int tls;
float lane, distance;
int numLane;
float occlusionThresh = 0.5;
float ambientLightComponent;
float lightX, lightZ;
ArrayList<BoundingBox> boundingBoxes;
ArrayList<String> classes;
ArrayList<Integer> blackCarChildren;
boolean lightsOn;
boolean context = true, contextOnly = false, noContextOnly = false;
float carW, carH, carD;
int startTime;
String frames_path = "frames", templates_path = "templates", labels_path = "labels";

void setup() {
  startTime = millis();
  if (args.length == 0) {
    nImages = 20000;
    println("No specification of the number of images to be generated. Generating" + String.valueOf(nImages) + "...");
  }
  if (args.length > 3)
    println("Two many input arguments. Ignoring from the fourth foward...");

  nImages = int(args[0]);
  if (args.length > 1) {
    if (args[1].equals("context"))
        contextOnly = true;
    else if (args[1].equals("no-context"))
        noContextOnly = true;
    else {
        if (isNumber(args[1]))
            seed = int(args[1]);
        else
            println("Argument not recognized: " + args[1] + ". Use \"context\" to generate contextualized data only," +
                " \"no-context\" to generate non-contextualized data only. Do not specify it to generate both.");
    }
    
    if (args.length == 3) {
        if (isNumber(args[2]))
            seed = int(args[2]);
        else
            println("Argument not recognized: " + args[2] + ". Adopting random seed " + String.valueOf(seed));
    }
  }
  
  if (seed != 1) {
    String strSeed = String.valueOf(seed);
    frames_path += strSeed;
    templates_path += strSeed;
    labels_path += strSeed;
  }
  
  //println(args[1]);
  //if (args[1].toString().equals("--no-context"))
  //  context = false;
  size(640, 480, P3D);
  textAlign(CENTER);
  textSize(30);
  //randomSeed(50);
  pg = createGraphics(width, height, P3D);
  car = loadShape("/home/jpvmello/sketchbook/traffic/7w2jvl6quxds-bmw_x5/bmw_x5/BMWX54.obj");
  carCopy = loadShape("/home/jpvmello/sketchbook/traffic/7w2jvl6quxds-bmw_x5/bmw_x5/BMWX54.obj");
  carW = carCopy.getWidth();
  carH = carCopy.getHeight();
  carD = carCopy.getDepth();
  classes = new ArrayList<String>();
  blackCarChildren = new ArrayList<Integer>();
  classes.add("red");  classes.add("green");  classes.add("yellow");
  blackCarChildren.add(0); blackCarChildren.add(2); blackCarChildren.add(4);
  blackCarChildren.add(6); blackCarChildren.add(8); blackCarChildren.add(9); 
  blackCarChildren.add(15); blackCarChildren.add(16); blackCarChildren.add(17);
  blackCarChildren.add(21);
  for (Integer child: blackCarChildren) {
    PShape cCopy = carCopy.getChild(child);
    cCopy.setFill(color(0));
  }
  initialSeed = seed;
  
  println();
  println("Number of images to be generated: " + String.valueOf(nImages));
  print("Generating contextualized set: ");
  if (!noContextOnly)
    println("Yes.");
  else
    println("No.");
  print("Generating non-contextualized set: ");
  if (!contextOnly)
    println("Yes.");
  else
    println("No.");
  println("Initial random seed: " + String.valueOf(seed));
  println("Random seed value is incremented when a new frame is generated or discarded due to absence" + 
    " of eligible traffic lights. Final seed value will be exhibit in the end of the run and saved in a separated file." + 
    " Be careful not to generate repeated scenes in different sets if not desired.");
}

void draw() {
    background(0);
    fill(255);
    text("Processing image " + nImage + " of " + nImages + "...", width/2, height/2);
  
    if (contextOnly)
        context = true;
    if (noContextOnly)
        context = false;
  
    do {
        tls = numLane = 0;
        boundingBoxes = new ArrayList<BoundingBox>();
        randomSeed(seed);
        pg.beginDraw();
          ambientLightComponent = random(256);
          lightX = random(-1, 1);
          lightZ = random(-1, 1);
          setLightsOn(true);
          pg.rectMode(CENTER);
          pg.perspective(PI/3.0, (float) width/height, 1, 100000);
          road = new Road();
          lane = road.randomRightLane();
          distance = random(8.9, 21)*height;
          pg.camera(lane, distance, random(200, 300), lane, height/2, 0, 0, 0, -1);
          pg.background(255, 255, 255, 0);
          pg.translate(width/2, height/2);
          road.draw();
          if (context) 
            pg.save(frames_path + "/frame_" + String.valueOf(nImage) + ".png");
          else
            pg.save(templates_path + "/frame_" + String.valueOf(nImage) + ".png");
        pg.endDraw();
        context = !context;
    } while (!context && !contextOnly && !noContextOnly);
    if (boundingBoxes.size() > 0) {
      PrintWriter labelsFile = createWriter(labels_path + "/frame_" + String.valueOf(nImage) + ".txt");
      for (BoundingBox boundingBox: boundingBoxes) {
        labelsFile.println(
          boundingBox.getCls() + " " + String.valueOf(boundingBox.getNormX()) + " " + String.valueOf(boundingBox.getNormY())
           + " " + String.valueOf(boundingBox.getNormWidth()) + " " + String.valueOf(boundingBox.getNormHeight())
        );
      }
      labelsFile.flush();
      labelsFile.close();
  
      nImage++;
      if (nImage > nImages) {
        String strInitialSeed = String.valueOf(initialSeed);
        String strSeed = String.valueOf(seed);
        String seeding = "Initial seed: " + strInitialSeed + "\nFinal seed: " + strSeed;
        PrintWriter seedLog = createWriter("initial_seed_" + strInitialSeed + "_final_seed_" + strSeed + ".txt");
        seedLog.println(seeding);
        seedLog.flush();
        seedLog.close();
        println(seeding);
        println("Total time:", time());
        exit();
      }
    }
    seed++;
}

class BoundingBox {
  String cls;
  float x1, y1, x2, y2;
  
  public BoundingBox(float x1, float y1, float x2, float y2) {
    this.cls = (String) classes.get(int(random(0, 3)));
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
  }
  
  public float getX1() {
    return x1;
  }
  
  public float getY1() {
    return y1;
  }
  
  public float getX2() {
    return x2;
  }
  
  public float getY2() {
    return y2;
  }
  
  public String getCls() {
    return cls;
  }
  
  public float getArea() {
      return (x2 - x1) * (y2 - y1);
  }
  
  public float getNormX() {
    return 0.5*(x1 + x2)/width;
  }
  
  public float getNormY() {
    return 0.5*(y1 + y2)/height;
  }
  
  public float getNormWidth() {
    return (x2 - x1)/width;
  }
  
  public float getNormHeight() {
    return (y2 - y1)/height;
  }
}
