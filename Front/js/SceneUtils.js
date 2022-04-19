import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export function initCamera(scene)
{
    scene.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 2000 );
    //scene.camera.position.z = 25;
    const pointLight = new THREE.PointLight( 0xffffff, 0.8 );
    scene.camera.add( pointLight );
    scene.scene.add( scene.camera );
} 

export function initLighting(scene)
{
    const hemiLight = new THREE.HemisphereLight( 0xffffff, 0x444444 );
    hemiLight.position.set( 0, 20, 0 );
    scene.env.add( hemiLight );

    const dirLight = new THREE.DirectionalLight( 0xffffff );
    dirLight.position.set( 3, 10, 10 );
    dirLight.castShadow = true;
    dirLight.shadow.camera.top = 2;
    dirLight.shadow.camera.bottom = - 2;
    dirLight.shadow.camera.left = - 2;
    dirLight.shadow.camera.right = 2;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 40;
    scene.env.add( dirLight );
    
}

export function initRenderer(scene)
{
    scene.renderer = new THREE.WebGLRenderer({ antialias: true});
    scene.renderer.setPixelRatio( window.devicePixelRatio );
    scene.renderer.setSize( window.innerWidth, window.innerHeight );
    scene.container.appendChild( scene.renderer.domElement );
}

export function initControls(scene)
{
    const controls = new OrbitControls( scene.camera, scene.renderer.domElement );
    controls.zoomSpeed = 3
    controls.target.set( 0, 1, 0 ); 
    controls.update();
}

export function initGround(scene)
{
    const mesh = new THREE.Mesh( new THREE.PlaneGeometry( 1000, 1000 ), new THREE.MeshPhongMaterial( { color: 0x999999, depthWrite: false } ) );
    mesh.rotation.x = - Math.PI / 2;
    mesh.receiveShadow = true;
    scene.env.add( mesh );
    
    scene.scene.background = new THREE.Color( 0xa0a0a0 );
    scene.scene.fog = new THREE.Fog( 0xa0a0a0, 10, 500 );
}
