import * as THREE from 'three';
import * as SceneUtils from './SceneUtils.js'

class Scene {
  constructor() {
    this.container = document.createElement('div');
    this.camera = null
    this.renderer = null
    this.env = new THREE.Group()
    this.scene = new THREE.Scene
    this.init();
  }

  init() {
    document.body.appendChild(this.container);
    SceneUtils.initCamera(this);
    SceneUtils.initLighting(this);
    SceneUtils.initCamera(this);
    SceneUtils.initRenderer(this);
    SceneUtils.initGround(this);
    SceneUtils.initControls(this);
  }

  initEventListeners()
  {
    this.onWindowResize()
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  animate() {
    requestAnimationFrame(this.animate);
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}

export {Scene}
