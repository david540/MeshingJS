import * as THREE from 'three';
import * as SceneUtils from './SceneUtils.js'
import { GuiPanel } from './panel.js'
import { Mesh } from './mesh.js'

class Scene {
  constructor() {
    this.container = document.createElement('div');
    this.camera = null
    this.renderer = null
    this.env = new THREE.Group()
    this.scene = new THREE.Scene
    this.mesh = new Mesh();
    this.raycaster = new THREE.Raycaster();
    //this.raycaster.params.Line.threshold = 0.000001;
    this.raycaster.params.Points.threshold = 0.1;
    this.pointer = new THREE.Vector2(0, 0);
    this.gui = null
    this.selVert = null;
    this.init();
  }

  async init() {
    document.body.appendChild(this.container);
    SceneUtils.initCamera(this);
    SceneUtils.initLighting(this);
    SceneUtils.initCamera(this);
    SceneUtils.initRenderer(this);
    SceneUtils.initGround(this);
    SceneUtils.initControls(this);
    this.initEventListeners();
    await this.mesh.loadMesh(this.camera);
    this.scene.add(this.mesh.object)
    this.gui = new GuiPanel(this)
    this.gui.init()
  }

  onPointerMove( event ) {
    this.pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    this.pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
  }

  initEventListeners() {
    window.addEventListener('resize', this.onWindowResize.bind(this));
    document.addEventListener( 'pointermove', this.onPointerMove.bind(this) );
    document.addEventListener('keyup', this.onDocumentKeyDown.bind(this));
    const fileInput = document.getElementById('file-upload-input');
    fileInput.onchange = (() => {
      const selectedFile = fileInput.files[0];
      this.mesh.filePath = URL.createObjectURL(selectedFile)
      this.scene.remove(this.mesh.object)
      this.mesh.loadMesh(this.camera)
      this.scene.add(this.mesh.object)
      //this.scene.add(this.mesh.pointsCloud)
    }).bind(this)
  }

  onDocumentKeyDown(event) {
    console.log("Ho !");
    switch (event.keyCode) {
      case 68:
        if(this.selVert !== null) {
          const attr = this.mesh.pointsCloud.geometry.attributes;
          attr.singu_valence.array[ this.selVert ] --;
          attr.singu_valence.needsUpdate = true;
          break;
        }
        console.log("Singu valence downgraded")
        break;
      case 70:
        if(this.selVert !== null) {
          const attr = this.mesh.pointsCloud.geometry.attributes;
          attr.singu_valence.array[ this.selVert ] ++;
          attr.singu_valence.needsUpdate = true;
          break;
        }
        console.log("Singu valence upgraded")
        break;
    }
  };

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.render();
  }
  

  render() {
    this.renderer.render(this.scene, this.camera);
    this.raycaster.setFromCamera( this.pointer, this.camera );
    if(this.mesh.computation_mesh && this.gui.displayOptions.bSinguEdit){
      //console.log("Hello ! :)")
      //console.log(this.pointer, this.camera);
      var intersects = this.raycaster.intersectObjects( this.mesh.pointGroup.children, true );//this.raycaster.intersectObject( this.mesh.object, true );
      //console.log(intersects);
      var flag = false;
      if ( intersects.length > 0 ) {
        for(var i = 0; i < intersects.length; i++) if(!flag && intersects[i].index && this.selVert != intersects[i].index) {
          console.log(intersects[i].index)//if(intersects[ i ].isPoints) console.log(intersects[ i ].index)
          const attr = this.mesh.pointsCloud.geometry.attributes;
          if(this.selVert !== null ) attr.size.array[ this.selVert ] /= 2;
          this.selVert = intersects[i].index;
          attr.size.array[ this.selVert ] *= 2;//= 30. / Math.sqrt(attr.size.count);
          attr.size.needsUpdate = true;
          flag = true;
          break;
        }
      }if(!flag &&  this.selVert !== null){
        /*const attr = this.mesh.pointsCloud.geometry.attributes;
        attr.size.array[ this.selVert ] /= 2; // 10. / Math.sqrt(attr.size.count);
				attr.size.needsUpdate = true;
				this.selVert = null;*/
      }
    }
  }
}

export { Scene }
