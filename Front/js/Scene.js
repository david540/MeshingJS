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
    this.init();
    this.gui = null
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
    await this.mesh.loadMesh();
    this.scene.add(this.mesh.object)
    this.gui = new GuiPanel(this)
    this.gui.init()
  }

  initEventListeners() {
    window.addEventListener('resize', this.onWindowResize.bind(this));
    const fileInput = document.getElementById('file-upload-input');
    fileInput.onchange = (() => {
      const selectedFile = fileInput.files[0];
      this.mesh.filePath = URL.createObjectURL(selectedFile)
      this.scene.remove(this.mesh.object)
      this.mesh.loadMesh()
      this.scene.add(this.mesh.object)
    }).bind(this)
  }

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
  }
}

export { Scene }
