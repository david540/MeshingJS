import * as THREE from 'three';
import { SingletonPanel } from './panel.js'
import { MeshOpe } from './mesh.js'
import { OBJLoader } from './three.js/examples/jsm/loaders/OBJLoader.js';
import { OrbitControls } from './three.js/examples/jsm/controls/OrbitControls.js';

let container;
let camera, scene, renderer;

init();
animate();

function init() {

    container = document.createElement( 'div' );
    document.body.appendChild( container );

    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 2000 );
    camera.position.z = 25;
    //camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 100 );
	//camera.position.set( - 1, 2, 3 );

    scene = new THREE.Scene();

    scene.background = new THREE.Color( 0xa0a0a0 );
	   scene.fog = new THREE.Fog( 0xa0a0a0, 10, 500 );

    const hemiLight = new THREE.HemisphereLight( 0xffffff, 0x444444 );
    hemiLight.position.set( 0, 20, 0 );
    scene.add( hemiLight );

   // const ambientLight = new THREE.AmbientLight( 0xcccccc, 0.4 );
   // scene.add( ambientLight );

    const pointLight = new THREE.PointLight( 0xffffff, 0.8 );
    camera.add( pointLight );
    scene.add( camera );

    const dirLight = new THREE.DirectionalLight( 0xffffff );
    dirLight.position.set( 3, 10, 10 );
    dirLight.castShadow = true;
    dirLight.shadow.camera.top = 2;
    dirLight.shadow.camera.bottom = - 2;
    dirLight.shadow.camera.left = - 2;
    dirLight.shadow.camera.right = 2;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 40;
    scene.add( dirLight );

    // ground

    const mesh = new THREE.Mesh( new THREE.PlaneGeometry( 1000, 1000 ), new THREE.MeshPhongMaterial( { color: 0x999999, depthWrite: false } ) );
    mesh.rotation.x = - Math.PI / 2;
    mesh.receiveShadow = true;
    scene.add( mesh );


    var object;
    var meshOpe;

    const manager = new THREE.LoadingManager();
    const loader = new OBJLoader( manager );

    loader.load( './js/three.js/examples/models/obj/emerald.obj', function ( obj ) {
        object = obj;
        object.position.x = 0; object.position.y = 0; object.position.z = 0;
        const mesh = object.children[0].clone();
        meshOpe = new MeshOpe(mesh);
        SingletonPanel.init(scene, meshOpe);
        meshOpe.display(scene);
    });

    document.querySelector('.inputfile').addEventListener('change', function(e){
        var dae_path;
        var files = e.currentTarget.files;
        if(files.length < 1) return;
        var extraFiles = {}, file;
        for (var i = 0; i < 1; i++){//files.length; i++) {
          file = files[i];
          extraFiles[file.name] = file;
          dae_path = file.name;
        }
        const manager = new THREE.LoadingManager();
        manager.setURLModifier(function (url, path) {
          url = url.split('/');
          url = url[url.length - 1];
          if (extraFiles[url] !== undefined) {
            var blobURL = URL.createObjectURL(extraFiles[url]);
            console.log(blobURL); //Blob location created from files selected from file input
            return blobURL;
          }return url;
        });
        const loader = new OBJLoader( manager );
        loader.load(dae_path, function ( obj ) {
            object = obj;
            scene.remove(scene.children[scene.children.length - 1]);
            object.position.x = 0; object.position.y = 0; object.position.z = 0;
            const mesh = object.children[0].clone();
            meshOpe.reset(mesh);
            meshOpe.display(scene);
        });
    });

    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    container.appendChild( renderer.domElement );
    const controls = new OrbitControls( camera, renderer.domElement );
    controls.target.set( 0, 1, 0 ); controls.update();
    window.addEventListener( 'resize', onWindowResize );
}

function onWindowResize() {
    const windowHalfX = window.innerWidth / 2;
    const windowHalfY = window.innerHeight / 2;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
}
function animate() {
    requestAnimationFrame( animate );
    render();
}
function render() {
    renderer.render( scene, camera );
}
