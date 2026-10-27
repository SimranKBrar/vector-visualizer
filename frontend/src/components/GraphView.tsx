import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import type { Dim, Vector } from '../types'

export const WORLD_SCALE = 10 // coords are normalised to [-1, 1]; this maps them into world units

interface Props {
  vectors: Vector[]
  enabledIds: Set<string>
  coords: Record<string, number[]> | null
  dim: Dim
  showLines: boolean
  showLabels: boolean
  selectedId: string | null
  hoverId: string | null
  neighborIds: Set<string>
  /** id -> rank (1 = closest) for the selected vector's top neighbours */
  neighborRanks: Map<string, number>
  /** key `${a}|${b}` (ids sorted) -> true embedding distance for the active metric */
  distanceLookup: Map<string, number>
  onSelect: (id: string | null) => void
  onHover: (id: string | null) => void
  resetKey: number
  /** bump to frame the camera on the selected vector + its neighbours */
  focusKey: number
  /** ceiling for on-screen dot size, in CSS px (toolbar slider) */
  dotMaxPx?: number
  /** fired on double-click of a point, so the caller can focus that grouping */
  onRequestFocus?: () => void
  /** called with a human-readable hint, e.g. too many distance labels */
  onHint?: (hint: string | null) => void
}

const MAX_DISTANCE_LABELS = 90
const MAX_POINT_LABELS = 60

function pairKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`
}

function fmt(x: number): string {
  const ax = Math.abs(x)
  if (ax >= 1000) return x.toFixed(0)
  if (ax >= 1) return x.toFixed(2)
  return x.toFixed(3)
}

function hashHue(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360
  return h
}

function labelFor(v: Vector): string {
  const meta = v.metadata ?? {}
  const named = (meta.label ?? meta.title ?? meta.name) as string | undefined
  const text = named ? `${v.id} · ${named}` : v.id
  return text.length > 34 ? `${text.slice(0, 33)}…` : text
}

function makeLabel(
  text: string,
  color: string,
  bg: string,
  targetPx: number,
  maxWorld: number,
  accent?: string, // neon HUD border + glow in this colour
): THREE.Sprite {
  const fontSize = 44
  const padX = 16
  const padY = 10
  const font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, sans-serif`

  const measure = document.createElement('canvas').getContext('2d')!
  measure.font = font
  const w = Math.ceil(measure.measureText(text).width) + padX * 2
  const h = fontSize + padY * 2

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.font = font
  const radius = 12
  const roundRect = () => {
    ctx.beginPath()
    ctx.moveTo(radius, 0)
    ctx.lineTo(w - radius, 0)
    ctx.quadraticCurveTo(w, 0, w, radius)
    ctx.lineTo(w, h - radius)
    ctx.quadraticCurveTo(w, h, w - radius, h)
    ctx.lineTo(radius, h)
    ctx.quadraticCurveTo(0, h, 0, h - radius)
    ctx.lineTo(0, radius)
    ctx.quadraticCurveTo(0, 0, radius, 0)
    ctx.closePath()
  }
  // neon halo bleeding out from behind the chip
  if (accent) {
    ctx.shadowColor = accent
    ctx.shadowBlur = 9
  }
  ctx.fillStyle = bg
  roundRect()
  ctx.fill()
  ctx.shadowBlur = 0
  if (accent) {
    // glassmorphic top sheen
    const glass = ctx.createLinearGradient(0, 0, 0, h)
    glass.addColorStop(0, 'rgba(255,255,255,0.13)')
    glass.addColorStop(0.55, 'rgba(255,255,255,0)')
    ctx.fillStyle = glass
    roundRect()
    ctx.fill()
    // hairline border in the accent colour (thick in canvas px because the
    // sprite renders at ~1/3 of canvas size — 3.5 ≈ 1.1 screen px)
    ctx.globalAlpha = 0.6
    ctx.strokeStyle = accent
    ctx.lineWidth = 3.5
    roundRect()
    ctx.stroke()
    ctx.globalAlpha = 1
  }
  ctx.font = font
  ctx.fillStyle = color
  ctx.textBaseline = 'middle'
  ctx.fillText(text, padX, h / 2 + 2)

  const tex = new THREE.CanvasTexture(canvas)
  tex.minFilter = THREE.LinearFilter
  tex.colorSpace = THREE.SRGBColorSpace
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true })
  const sprite = new THREE.Sprite(mat)
  sprite.scale.set(maxWorld * (w / h), maxWorld, 1)
  sprite.renderOrder = 20
  // sizing is recomputed every frame: keep labels ~targetPx on screen while
  // zoomed in, capped at maxWorld world-units so they never swallow the graph
  sprite.userData = { aspect: w / h, targetPx, maxWorld, text }
  return sprite
}

function clearGroup(group: THREE.Group) {
  group.traverse((obj) => {
    const mesh = obj as THREE.Mesh
    if (mesh.geometry && !mesh.geometry.userData.shared) mesh.geometry.dispose()
    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(mat)) mat.forEach((m) => disposeMaterial(m))
    else if (mat) disposeMaterial(mat)
  })
  group.clear()
}

function disposeMaterial(m: THREE.Material) {
  const map = (m as THREE.MeshBasicMaterial).map
  // shared textures (cosmic sprites) outlive the material that samples them
  if (map && !map.userData.shared) map.dispose()
  m.dispose()
}

function pointRadius(n: number): number {
  return Math.min(1, Math.max(0.3, 0.95 * Math.pow(40 / Math.max(n, 1), 0.35)))
}

// ---- "galaxy" visuals ---------------------------------------------------
// Nebula-hued palette: every cluster is tinted a cosmic colour.
// Deliberately excludes the two state colours — amber (selected) and mint
// (nearest neighbours) — so a tinted body never reads as a highlighted one
const COSMOS = ['#a3e635', '#54d2c0', '#b78cff', '#ff9ecb', '#ff9f7a', '#8fd3ff']

// Each vector is drawn as a little cosmic body — one of four procedural
// shapes (spiral galaxy / radiant star / nebula star-cluster / binary star).
// All are drawn pure white so SpriteMaterial.color can tint them (cluster /
// neighbour / selected hues) and their cores glow against the nebula.
function makeCosmicSprite(seed: number): THREE.CanvasTexture {
  const S = 160
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = S
  const ctx = canvas.getContext('2d')!
  // deterministic jitter so every variant stays stable across rebuilds
  let s = (seed * 16807 + 11) % 2147483647
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647)
  const cx = S / 2
  const cy = S / 2
  const white = (a: number) => `rgba(255,255,255,${a})`

  const softGlow = (x: number, y: number, r: number, a: number) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, white(a))
    g.addColorStop(1, white(0))
    ctx.fillStyle = g
    ctx.fillRect(0, 0, S, S)
  }
  const hotCore = (x: number, y: number, r: number) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, white(1))
    g.addColorStop(0.45, white(0.9))
    g.addColorStop(1, white(0))
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fill()
  }
  // 4-point diffraction spikes, as through a telescope
  const spikes = (x: number, y: number, len: number, w2: number, a: number, spin: number) => {
    for (let k = 0; k < 4; k++) {
      const ang = spin + (k * Math.PI) / 2
      const diag = k % 2 === 1
      ctx.save()
      ctx.translate(x, y)
      ctx.rotate(ang)
      const g = ctx.createLinearGradient(0, 0, len, 0)
      g.addColorStop(0, white(a))
      g.addColorStop(1, white(0))
      ctx.fillStyle = g
      ctx.beginPath()
      ctx.moveTo(0, -w2)
      ctx.lineTo(diag ? len * 0.55 : len, 0)
      ctx.lineTo(0, w2)
      ctx.closePath()
      ctx.fill()
      ctx.restore()
    }
  }

  softGlow(cx, cy, S * 0.5, 0.18) // ambient halo on every body
  const kind = Math.abs(seed) % 4

  if (kind === 0) {
    // ---- face-on spiral galaxy ------------------------------------
    const turns = 1.7 + rnd() * 0.7
    const squash = 0.62 + rnd() * 0.3 // tilt: squash the disc slightly
    const spin = rnd() * Math.PI * 2
    for (let arm = 0; arm < 2; arm++) {
      const dots = 95
      for (let k = 0; k < dots; k++) {
        const t = k / (dots - 1)
        const th = t * turns * Math.PI * 2
        const rr = S * 0.055 + t * S * 0.4
        const jitter = (rnd() - 0.5) * S * 0.055
        const ang = spin + arm * Math.PI + th
        const x = cx + Math.cos(ang) * (rr + jitter)
        const y = cy + Math.sin(ang) * (rr + jitter) * squash
        const a = (1 - t * 0.75) * (0.35 + rnd() * 0.55)
        ctx.fillStyle = white(Math.max(0.06, a))
        ctx.beginPath()
        ctx.arc(x, y, S * (0.005 + (1 - t) * 0.009 + rnd() * 0.004), 0, Math.PI * 2)
        ctx.fill()
      }
    }
    hotCore(cx, cy, S * 0.11)
    ctx.fillStyle = white(1)
    ctx.beginPath()
    ctx.arc(cx, cy, S * 0.022, 0, Math.PI * 2)
    ctx.fill()
  } else if (kind === 1) {
    // ---- radiant star: hot core + diffraction spikes ---------------
    softGlow(cx, cy, S * 0.36, 0.4)
    spikes(cx, cy, S * 0.47, S * 0.02, 0.75, (rnd() - 0.5) * 0.35)
    hotCore(cx, cy, S * 0.13)
    // a couple of stray sparkles
    for (let k = 0; k < 3; k++) {
      ctx.fillStyle = white(0.35 + rnd() * 0.4)
      ctx.beginPath()
      ctx.arc(S * (0.16 + rnd() * 0.68), S * (0.16 + rnd() * 0.68), S * 0.006, 0, Math.PI * 2)
      ctx.fill()
    }
  } else if (kind === 2) {
    // ---- nebula star-cluster (stellar nursery) ---------------------
    const clouds = 3 + Math.floor(rnd() * 2)
    for (let k = 0; k < clouds; k++) {
      softGlow(cx + (rnd() - 0.5) * S * 0.34, cy + (rnd() - 0.5) * S * 0.34, S * (0.16 + rnd() * 0.16), 0.2 + rnd() * 0.14)
    }
    const stars = 5 + Math.floor(rnd() * 5)
    for (let k = 0; k < stars; k++) {
      const x = cx + (rnd() - 0.5) * S * 0.5
      const y = cy + (rnd() - 0.5) * S * 0.5
      softGlow(x, y, S * (0.03 + rnd() * 0.03), 0.6)
      hotCore(x, y, S * (0.012 + rnd() * 0.016))
    }
  } else {
    // ---- binary star system with a shared orbit --------------------
    const d = S * (0.13 + rnd() * 0.06)
    const tilt = rnd() * Math.PI
    const bx = cx + Math.cos(tilt) * d
    const by = cy + Math.sin(tilt) * d
    const sx = cx - Math.cos(tilt) * d
    const sy = cy - Math.sin(tilt) * d
    // faint elliptical orbit
    ctx.strokeStyle = white(0.3)
    ctx.lineWidth = S * 0.006
    ctx.beginPath()
    ctx.ellipse(cx, cy, d * 1.5, d * 0.55, tilt, 0, Math.PI * 2)
    ctx.stroke()
    softGlow(bx, by, S * 0.24, 0.5)
    softGlow(sx, sy, S * 0.16, 0.4)
    spikes(bx, by, S * 0.3, S * 0.014, 0.7, tilt)
    hotCore(bx, by, S * 0.085)
    hotCore(sx, sy, S * 0.05)
  }

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

// shared across rebuilds — must not be disposed with the materials that use it
let cosmicTexCache: THREE.CanvasTexture[] | null = null
function cosmicTexture(variant: number): THREE.CanvasTexture {
  if (!cosmicTexCache) {
    cosmicTexCache = [0, 1, 2, 3].map((seed) => {
      const t = makeCosmicSprite(seed)
      t.userData.shared = true
      return t
    })
  }
  return cosmicTexCache[Math.abs(variant) % cosmicTexCache.length]
}

// soft radial glow used for the galactic core behind the graph
let glowTexCache: THREE.CanvasTexture | null = null
function glowTexture(): THREE.CanvasTexture {
  if (!glowTexCache) {
    const S = 256
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = S
    const ctx = canvas.getContext('2d')!
    const g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2)
    g.addColorStop(0, 'rgba(255,255,255,0.85)')
    g.addColorStop(0.3, 'rgba(255,255,255,0.32)')
    g.addColorStop(1, 'rgba(255,255,255,0)')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, S, S)
    glowTexCache = new THREE.CanvasTexture(canvas)
    glowTexCache.userData.shared = true
  }
  return glowTexCache
}

// deep-space backdrop: layered nebulae + baked star dust
function makeNebulaTexture(): THREE.CanvasTexture {
  const w = 1024
  const h = 512
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#030309'
  ctx.fillRect(0, 0, w, h)

  let s = 987654321
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647)

  // same-hue transparent stops avoid dark fringes when gradients overlap
  const cloud = (x: number, y: number, r: number, rgb: [number, number, number], a: number) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`)
    g.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`)
    ctx.fillStyle = g
    ctx.fillRect(0, 0, w, h)
  }
  cloud(512, 250, 430, [96, 64, 190], 0.2) // violet body
  cloud(512, 250, 260, [170, 96, 220], 0.16)
  cloud(512, 248, 130, [255, 214, 170], 0.2) // warm galactic core
  cloud(170, 130, 330, [56, 96, 220], 0.14) // blue wing
  cloud(860, 380, 340, [40, 190, 190], 0.11) // teal wing
  cloud(140, 440, 260, [180, 70, 220], 0.13)
  cloud(900, 90, 250, [70, 80, 230], 0.12)

  // baked star dust
  for (let i = 0; i < 750; i++) {
    const x = rnd() * w
    const y = rnd() * h
    const m = rnd()
    const tint = m > 0.94 ? '186,214,255' : m > 0.88 ? '255,220,170' : '255,255,255'
    ctx.fillStyle = `rgba(${tint},${0.15 + rnd() * 0.55})`
    const r = m > 0.97 ? 1.4 + rnd() * 1.2 : 0.35 + rnd() * 0.75
    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fill()
  }

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

// drifting 3-D starfield surrounding the whole scene
function makeStarfield(): THREE.Points {
  const N = 1500
  const pos = new Float32Array(N * 3)
  const col = new Float32Array(N * 3)
  for (let i = 0; i < N; i++) {
    const u = Math.random() * 2 - 1
    const th = Math.random() * Math.PI * 2
    const rr = Math.sqrt(1 - u * u)
    const rad = 55 + Math.random() * 300
    pos[i * 3] = rad * rr * Math.cos(th)
    pos[i * 3 + 1] = rad * u
    pos[i * 3 + 2] = rad * rr * Math.sin(th)
    const t = Math.random()
    let rgb: [number, number, number] = [1, 1, 1]
    if (t > 0.92) rgb = [1, 0.66, 0.88]
    else if (t > 0.82) rgb = [1, 0.84, 0.55]
    else if (t > 0.68) rgb = [0.62, 0.8, 1]
    const k = 0.45 + Math.random() * 0.55
    col[i * 3] = rgb[0] * k
    col[i * 3 + 1] = rgb[1] * k
    col[i * 3 + 2] = rgb[2] * k
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
  const mat = new THREE.PointsMaterial({
    size: 1.1,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: 0.75,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  return new THREE.Points(geo, mat)
}

// faint orbital rings marking the galactic plane (3D only)
function makeGalaxyRings(): THREE.Group {
  const group = new THREE.Group()
  const rings: [number, number][] = [
    [13, 0.2],
    [21, 0.14],
    [29, 0.09],
  ]
  for (const [radius, opacity] of rings) {
    const pts: THREE.Vector3[] = []
    for (let i = 0; i <= 180; i++) {
      const a = (i / 180) * Math.PI * 2
      pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius))
    }
    group.add(
      new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: '#8b7cf6', transparent: true, opacity }),
      ),
    )
  }
  return group
}

interface SceneState {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  pointsGroup: THREE.Group
  linesGroup: THREE.Group
  labelsGroup: THREE.Group
  grid: THREE.Group // orbital rings marking the galactic plane
  raycaster: THREE.Raycaster
  pointer: THREE.Vector2
  resizeObserver: ResizeObserver
  props: Props
}

export default function GraphView(props: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef<SceneState | null>(null)
  const propsRef = useRef(props)
  propsRef.current = props

  // ---- one-time scene setup -------------------------------------------
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.domElement.style.display = 'block'
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = makeNebulaTexture() // deep-space nebula backdrop

    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000)
    camera.position.set(0, 0, 34)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.screenSpacePanning = true
    // wheel-zoom moves toward the cursor, so you zoom *into a grouping*
    // instead of dollying at the graph's origin
    controls.zoomToCursor = true
    controls.minDistance = 0.5
    controls.maxDistance = 300

    const pointsGroup = new THREE.Group()
    const linesGroup = new THREE.Group()
    const labelsGroup = new THREE.Group()
    scene.add(pointsGroup, linesGroup, labelsGroup)

    // a faint galactic-core glow behind the graph; additive + drawn first so
    // everything else composites over it
    const coreGlow = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: glowTexture(),
        color: '#8f7bff',
        transparent: true,
        opacity: 0.2,
        blending: THREE.AdditiveBlending,
        depthTest: false,
        depthWrite: false,
      }),
    )
    coreGlow.scale.set(70, 70, 1)
    coreGlow.renderOrder = -1
    scene.add(coreGlow)

    // bloom post-processing: cores, halos, stars and the nebula's heart all
    // glow — threshold keeps the dark label chips and backdrop crisp
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    composer.addPass(
      new UnrealBloomPass(
        new THREE.Vector2(container.clientWidth || 1, container.clientHeight || 1),
        0.32, // strength
        0.4, // radius
        0.8, // threshold: only the hottest cores bloom
      ),
    )
    composer.addPass(new OutputPass())

    // drifting starfield around everything (never pickable: picking only
    // raycasts pointsGroup) + faint orbital rings for the galactic plane
    const starField = makeStarfield()
    const starMat = starField.material as THREE.PointsMaterial
    scene.add(starField)

    const grid = makeGalaxyRings()
    grid.visible = false
    scene.add(grid)

    // ---- navigation cube (view gizmo) ----------------------------------
    // A small solid orientation cube rendered into the bottom-right corner
    // every frame; grab it and drag to orbit the camera (3D only), or click
    // an axis badge to snap the view to that direction.
    const GIZMO_SIZE = 92 // css px — must match .gizmo-panel in index.css
    const GIZMO_RIGHT = 14 // css px inset from the canvas' right edge
    const GIZMO_BOTTOM = 74 // css px from canvas bottom — clears the legend bar
    const GIZMO_DIRS: [string, THREE.Vector3][] = [
      ['+x', new THREE.Vector3(1, 0, 0)],
      ['-x', new THREE.Vector3(-1, 0, 0)],
      ['+y', new THREE.Vector3(0, 1, 0)],
      ['-y', new THREE.Vector3(0, -1, 0)],
      ['+z', new THREE.Vector3(0, 0, 1)],
      ['-z', new THREE.Vector3(0, 0, -1)],
    ]
    const AXES: [THREE.Vector3, number, string][] = [
      [new THREE.Vector3(1, 0, 0), 0xff6b6b, 'X'],
      [new THREE.Vector3(0, 1, 0), 0x6ee7b7, 'Y'],
      [new THREE.Vector3(0, 0, 1), 0x7aa2ff, 'Z'],
    ]
    const gizmoScene = new THREE.Scene()
    const gizmoCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 20)
    {
      // backdrop plane (square minus a hair, so the rounded .gizmo-panel
      // frame peeks around it); drawn first, behind the cube
      const bg = new THREE.Mesh(
        new THREE.PlaneGeometry(3.0, 3.0),
        new THREE.MeshBasicMaterial({ color: '#0a0716', transparent: true, opacity: 0.93, depthWrite: false }),
      )
      bg.name = 'gizmo-bg'
      bg.renderOrder = 0
      gizmoScene.add(bg)
      // solid, translucent faces so the gizmo reads as a real cube you can
      // grab; FrontSide so only the near three faces ever show, each face a
      // slightly different shade so orientation stays legible
      const face = (hex: string) =>
        new THREE.MeshBasicMaterial({ color: hex, transparent: true, opacity: 0.34, depthWrite: false })
      const faces = new THREE.Mesh(
        new THREE.BoxGeometry(2, 2, 2),
        // material order = +x, -x, +y, -y, +z, -z
        [face('#9d8fff'), face('#6f5fe0'), face('#c0b4ff'), face('#4d3fb0'), face('#8b7cf6'), face('#5a4ac9')],
      )
      faces.renderOrder = 1
      gizmoScene.add(faces)
      const box = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
        new THREE.LineBasicMaterial({ color: '#d6ccff', transparent: true, opacity: 1 }),
      )
      box.renderOrder = 2
      gizmoScene.add(box)
      for (const [dir, color, letter] of AXES) {
        const end = dir.clone().multiplyScalar(1.3)
        gizmoScene.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), end]),
            new THREE.LineBasicMaterial({ color }),
          ),
        )
        // the letter badge doubles as the clickable axis endpoint
        const badge = makeLabel(letter, '#0a0e17', `#${color.toString(16).padStart(6, '0')}`, 12, 1)
        badge.scale.set(0.46 * (badge.userData.aspect as number), 0.46, 1)
        badge.position.copy(end)
        badge.renderOrder = 30
        gizmoScene.add(badge)
      }
    }
    const gizmoPanel = document.createElement('div')
    gizmoPanel.className = 'gizmo-panel'
    container.appendChild(gizmoPanel)

    const state: SceneState = {
      renderer,
      scene,
      camera,
      controls,
      pointsGroup,
      linesGroup,
      labelsGroup,
      grid,
      raycaster: new THREE.Raycaster(),
      pointer: new THREE.Vector2(),
      resizeObserver: new ResizeObserver(() => {
        const w = container.clientWidth || 1
        const h = container.clientHeight || 1
        // refresh ratio (window may have moved across displays) and update the
        // canvas CSS size too — `updateStyle=false` leaves the canvas at its
        // device-pixel size, which overflows the container on HiDPI screens.
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
        renderer.setSize(w, h)
        composer.setSize(w, h)
        camera.aspect = w / h
        camera.updateProjectionMatrix()
      }),
      props,
    }
    stateRef.current = state
    // size immediately as well as on future resizes
    state.resizeObserver.observe(container)
    renderer.setSize(container.clientWidth || 1, container.clientHeight || 1)
    composer.setSize(container.clientWidth || 1, container.clientHeight || 1)

    // debug hook: lets tests (and users) inspect the scene / force a render
    ;(window as unknown as Record<string, unknown>).__vv = {
      renderer,
      scene,
      camera,
      controls,
      applyScreenScale: () => applyScreenScale(),
      composer,
      draw: () => {
        controls.update()
        applyScreenScale()
        composer.render()
        renderGizmo()
      },
      gizmoCamera,
      // absolute CSS-px position of a gizmo axis badge ('+x', '-y', …)
      gizmoEndpointPx: (sign: string) => {
        const dir = GIZMO_DIRS.find(([s]) => s === sign)?.[1]
        if (!dir) return null
        syncGizmoCamera()
        const p = dir.clone().multiplyScalar(1.3).project(gizmoCamera)
        const rect = renderer.domElement.getBoundingClientRect()
        return {
          x: rect.left + (rect.width - GIZMO_SIZE - GIZMO_RIGHT) + ((p.x + 1) / 2) * GIZMO_SIZE,
          y: rect.top + (rect.height - GIZMO_SIZE - GIZMO_BOTTOM) + ((1 - p.y) / 2) * GIZMO_SIZE,
        }
      },
    }

    // Re-derive on-screen sizes every frame so zooming in re-renders the view
    // for the new scale: labels keep a readable pixel size (capped in world
    // units so they never swallow the graph), and points never fall below a
    // visible screen size.
    const applyScreenScale = () => {
      const viewportH = renderer.domElement.clientHeight || 1
      const viewportW = renderer.domElement.clientWidth || 1
      const worldPerPixel = (distance: number) =>
        (2 * Math.tan(((camera.fov * Math.PI) / 180) / 2) * distance) / viewportH
      // refresh matrices: this runs from the debug hook / tests too, not just
      // after a render
      camera.updateMatrixWorld()
      camera.matrixWorldInverse.copy(camera.matrixWorld).invert()
      // Three sizes sprites from their *axial* depth (view-space z), not the
      // slant distance — use the same measure or labels near frame edges
      // render larger than we think and the declutter lets pairs through
      const axialDepth = (v: THREE.Vector3) =>
        Math.max(1e-3, -v.clone().applyMatrix4(camera.matrixWorldInverse).z)
      const projectXY = (v: THREE.Vector3): [number, number] => {
        const p = v.clone().project(camera)
        return [((p.x + 1) / 2) * viewportW, ((1 - p.y) / 2) * viewportH]
      }

      // size + re-anchor every label for this frame
      const sized: { sprite: THREE.Sprite; basePos: THREE.Vector3; w: number; h: number }[] = []
      for (const obj of labelsGroup.children) {
        const sprite = obj as THREE.Sprite
        // measure from the anchor: the sprite's own position carries the
        // previous frame's offset and would skew the size calculation
        const anchor = sprite.userData.anchor as THREE.Vector3 | undefined
        const dAnchor = axialDepth(anchor ?? sprite.position)
        const wanted = (sprite.userData.targetPx as number) * worldPerPixel(dAnchor)
        let h = Math.min(sprite.userData.maxWorld as number, Math.max(0.05, wanted))
        sprite.scale.set(h * (sprite.userData.aspect as number), h, 1)
        // keep the label hugging its anchor at every zoom level
        if (anchor) {
          const offset = h * ((sprite.userData.offsetK as number) || 0)
          sprite.position.set(anchor.x, anchor.y + offset, anchor.z)
          // refit at the sprite's own position: in oblique 3D the label sits
          // closer/farther than its anchor, so re-derive h once to land the
          // drawn size exactly on targetPx
          const h2 = Math.min(
            sprite.userData.maxWorld as number,
            Math.max(0.05, (sprite.userData.targetPx as number) * worldPerPixel(axialDepth(sprite.position))),
          )
          if (Math.abs(h2 - h) > 1e-6) {
            h = h2
            sprite.scale.set(h * (sprite.userData.aspect as number), h, 1)
            sprite.position.set(anchor.x, anchor.y + h * ((sprite.userData.offsetK as number) || 0), anchor.z)
          }
        }
        // rect measured exactly as the GPU draws it: sprite position, axial depth
        const d = axialDepth(sprite.position)
        sized.push({
          sprite,
          basePos: sprite.position.clone(),
          w: sprite.scale.x / worldPerPixel(d),
          h: h / worldPerPixel(d),
        })
      }

      // declutter: no two labels may overlap. Priority order — selected (0),
      // numbered neighbours (1-3), other neighbours (4), point labels (10),
      // distance labels (20). High-priority labels nudge out of the way first;
      // anything that still can't be placed without colliding is hidden.
      const items = sized.map((it) => {
        const viewZ = it.basePos.clone().applyMatrix4(camera.matrixWorldInverse).z
        const [cx, cy] = projectXY(it.basePos)
        return { ...it, priority: (it.sprite.userData.priority as number) ?? 50, cx, cy, viewZ }
      })
      items.sort((a, b) => a.priority - b.priority)
      const accepted: number[][] = []
      const intersects = (r: number[]) =>
        accepted.some((a) => r[0] < a[2] && r[2] > a[0] && r[1] < a[3] && r[3] > a[1])
      const offscreen = (r: number[]) =>
        r[2] < 0 || r[3] < 0 || r[0] > viewportW || r[1] > viewportH
      // world direction that maps to "up on screen" (the camera's own +Y axis)
      const screenUp = new THREE.Vector3()
      const scratch = new THREE.Vector3()
      const screenY = (v: THREE.Vector3) => {
        scratch.copy(v).project(camera)
        return ((1 - scratch.y) / 2) * viewportH
      }
      for (const it of items) {
        if (it.viewZ > -0.05) {
          it.sprite.visible = false // behind (or grazing) the camera
          continue
        }
        // ladder of vertical nudges, in screen px (0 first, then up/down steps)
        const steps = it.priority <= 10 ? [0, -1, 1, -2, 2, -3, 3, -4, 4] : [0]
        let placed = false
        for (const step of steps) {
          const dy = step * (it.h + 3)
          let pos = it.basePos
          let cx = it.cx
          let cy = it.cy + dy
          let w = it.w
          let h = it.h
          if (step !== 0) {
            // solve for the exact screen offset: nudge along camera-up,
            // correcting with the measured Jacobian (perspective compresses
            // world-Y motion away from the frame centre)
            screenUp.setFromMatrixColumn(camera.matrixWorld, 1)
            pos = it.basePos.clone()
            const targetY = it.cy + dy
            for (let k = 0; k < 3; k++) {
              const y0 = screenY(pos)
              scratch.copy(pos).addScaledVector(screenUp, 0.05)
              const jacobian = (y0 - screenY(scratch)) / 0.05 // px gained upward per world unit
              if (jacobian < 1e-6) break
              pos.addScaledVector(screenUp, (y0 - targetY) / jacobian)
              if (Math.abs(screenY(pos) - targetY) < 0.15) break
            }
            ;[cx, cy] = projectXY(pos)
            const d = axialDepth(pos)
            w = it.sprite.scale.x / worldPerPixel(d)
            h = it.sprite.scale.y / worldPerPixel(d)
          }
          const rect = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
          if (offscreen(rect) || intersects(rect)) continue
          accepted.push(rect)
          if (step !== 0) it.sprite.position.copy(pos)
          it.sprite.visible = true
          placed = true
          break
        }
        if (!placed) it.sprite.visible = false
      }
      // ---- bodies: envelope sizing, then a 2D anti-overlap pass ----------
      const bodies = pointsGroup.children.filter(
        (o) => o.userData.baseRadius !== undefined,
      ) as THREE.Sprite[]
      const MIN_PX = 3.2
      const MAX_PX = propsRef.current.dotMaxPx ?? 32 // user-controlled ceiling (toolbar)
      const targetPx = bodies.map((spr) => {
        const base = spr.userData.baseRadius as number
        // sprites render by axial depth (view-space z), same as labels
        const natural = base / worldPerPixel(axialDepth(spr.position))
        // keep bodies inside a screen-size envelope: never sub-pixel when far,
        // never ballooning into blobs when deep-zoomed — zoom separates the
        // grouping instead of swallowing it
        if (natural < MIN_PX) return Math.min(MIN_PX, natural * 4)
        if (natural > MAX_PX) return MAX_PX
        return natural
      })

      if (propsRef.current.dim === 2 && bodies.length > 1) {
        // flat plane: cap every dot to just under half its nearest-neighbour
        // screen distance so no two dots can overlap. Screen positions are
        // grid-bucketed so this stays O(n) even for large datasets.
        const CELL = 96 // > 2 * max dot px (slider tops out at 40)
        const sx: number[] = []
        const sy: number[] = []
        const buckets = new Map<string, number[]>()
        for (let i = 0; i < bodies.length; i++) {
          const p = bodies[i].position.clone().project(camera)
          sx.push(((p.x + 1) / 2) * viewportW)
          sy.push(((1 - p.y) / 2) * viewportH)
          const key = `${Math.floor(sx[i] / CELL)},${Math.floor(sy[i] / CELL)}`
          const list = buckets.get(key)
          if (list) list.push(i)
          else buckets.set(key, [i])
        }
        for (let i = 0; i < bodies.length; i++) {
          const cx = Math.floor(sx[i] / CELL)
          const cy = Math.floor(sy[i] / CELL)
          let nearestSq = Infinity
          for (let ox = -1; ox <= 1; ox++) {
            for (let oy = -1; oy <= 1; oy++) {
              const list = buckets.get(`${cx + ox},${cy + oy}`)
              if (!list) continue
              for (const j of list) {
                if (j === i) continue
                const ddx = sx[i] - sx[j]
                const ddy = sy[i] - sy[j]
                const d2 = ddx * ddx + ddy * ddy
                if (d2 < nearestSq) nearestSq = d2
              }
            }
          }
          if (nearestSq !== Infinity) {
            const cap = Math.sqrt(nearestSq) * 0.46 // just under half → guaranteed gap
            // non-overlap beats the min-size floor here: a dot may shrink
            // below 2.4px (0.3px epsilon keeps it alive) rather than overlap
            if (cap < targetPx[i]) targetPx[i] = Math.max(0.3, cap)
          }
        }
      }

      // targetPx is a half-width in screen px → sprite world size is 2× that
      for (let i = 0; i < bodies.length; i++) {
        const spr = bodies[i]
        const world = 2 * targetPx[i] * worldPerPixel(axialDepth(spr.position))
        spr.scale.set(world, world, 1)
      }
    }

    // point the gizmo camera along the same direction as the main camera
    const syncGizmoCamera = () => {
      const dir = camera.position.clone().sub(controls.target)
      if (dir.lengthSq() < 1e-9) dir.set(0, 0, 1)
      gizmoCamera.position.copy(dir).normalize().multiplyScalar(6)
      gizmoCamera.up.set(0, 1, 0)
      gizmoCamera.lookAt(0, 0, 0)
      gizmoCamera.updateMatrixWorld()
      // keep the backdrop directly behind the cube, facing the camera
      const bg = gizmoScene.getObjectByName('gizmo-bg')
      if (bg) {
        bg.position.copy(gizmoCamera.position).multiplyScalar(-0.85)
        bg.lookAt(gizmoCamera.position)
      }
    }

    // render the navigation cube into the bottom-right corner; it draws over
    // the graph (depth cleared, colour kept) so the scene shows through
    const renderGizmo = () => {
      const w = renderer.domElement.clientWidth || 1
      const h = renderer.domElement.clientHeight || 1
      syncGizmoCamera()
      const x = w - GIZMO_SIZE - GIZMO_RIGHT
      const y = GIZMO_BOTTOM // GL viewport y is measured from the bottom
      renderer.autoClear = false
      renderer.setViewport(x, y, GIZMO_SIZE, GIZMO_SIZE)
      renderer.setScissor(x, y, GIZMO_SIZE, GIZMO_SIZE)
      renderer.setScissorTest(true)
      renderer.clearDepth()
      renderer.render(gizmoScene, gizmoCamera)
      renderer.setScissorTest(false)
      renderer.setViewport(0, 0, w, h)
      renderer.autoClear = true
    }

    renderer.setAnimationLoop(() => {
      controls.update()
      // gentle galactic drift + breathing twinkle
      starField.rotation.y += 0.00025
      starMat.opacity = 0.5 + 0.28 * (0.5 + 0.5 * Math.sin(performance.now() * 0.0011))
      applyScreenScale()
      composer.render()
      renderGizmo()
    })

    // ---- pointer interaction ------------------------------------------
    const el = renderer.domElement
    let downX = 0
    let downY = 0

    const pick = (): string | null => {
      // matrices are normally refreshed by the render loop; update them here so
      // picking is correct even between frames (e.g. right after a camera move)
      camera.updateMatrixWorld()
      scene.updateMatrixWorld()
      const rect = el.getBoundingClientRect()
      state.pointer.x = ((lastX - rect.left) / rect.width) * 2 - 1
      state.pointer.y = -((lastY - rect.top) / rect.height) * 2 + 1
      state.raycaster.setFromCamera(state.pointer, camera)
      const hits = state.raycaster
        .intersectObjects(pointsGroup.children, true)
        .filter((h) => (h.object as THREE.Sprite).isSprite && h.object.userData.id !== undefined)
      return hits.length ? (hits[0].object.userData.id as string) : null
    }

    let lastX = 0
    let lastY = 0
    // pointer currently inside the gizmo square (3D only)
    const inGizmoSquare = (clientX: number, clientY: number) => {
      if (propsRef.current.dim !== 3) return false
      const rect = el.getBoundingClientRect()
      const lx = clientX - rect.left - (rect.width - GIZMO_SIZE - GIZMO_RIGHT)
      const ly = clientY - rect.top - (rect.height - GIZMO_SIZE - GIZMO_BOTTOM)
      return lx >= 0 && ly >= 0 && lx <= GIZMO_SIZE && ly <= GIZMO_SIZE
    }
    // grab the cube and drag → orbit the main camera; the cube mirrors the
    // camera every frame, so it turns with your drag and the view on screen
    // always matches the cube's faces
    let gizmoDrag = false
    let gizmoLastX = 0
    let gizmoLastY = 0
    const ORBIT_PER_PX = 0.006 // same feel as dragging the canvas
    const orbitByGizmoDrag = (dx: number, dy: number) => {
      const offset = camera.position.clone().sub(controls.target)
      const sph = new THREE.Spherical().setFromVector3(offset)
      sph.theta -= dx * ORBIT_PER_PX
      sph.phi = Math.min(Math.PI - 0.02, Math.max(0.02, sph.phi - dy * ORBIT_PER_PX))
      offset.setFromSpherical(sph)
      camera.position.copy(controls.target).add(offset)
      controls.update()
    }
    const onMove = (e: PointerEvent) => {
      if (gizmoDrag) {
        orbitByGizmoDrag(e.clientX - gizmoLastX, e.clientY - gizmoLastY)
        gizmoLastX = e.clientX
        gizmoLastY = e.clientY
        return
      }
      lastX = e.clientX
      lastY = e.clientY
      const id = pick()
      el.style.cursor = id ? 'pointer' : 'grab'
      propsRef.current.onHover(id)
    }
    const onDown = (e: PointerEvent) => {
      downX = e.clientX
      downY = e.clientY
      if (inGizmoSquare(e.clientX, e.clientY)) {
        // grab the cube: pause OrbitControls so the two don't fight over
        // the same drag; re-enabled in onUp
        gizmoDrag = true
        gizmoLastX = e.clientX
        gizmoLastY = e.clientY
        controls.enabled = false
        el.style.cursor = 'grabbing'
        el.setPointerCapture(e.pointerId)
      }
    }
    // click inside the navigation cube → nearest axis badge ('+x', '-y', …)
    const gizmoHit = (clientX: number, clientY: number): THREE.Vector3 | null => {
      if (propsRef.current.dim !== 3) return null // 2D stays locked top-down
      const rect = el.getBoundingClientRect()
      const lx = clientX - rect.left - (rect.width - GIZMO_SIZE - GIZMO_RIGHT)
      const ly = clientY - rect.top - (rect.height - GIZMO_SIZE - GIZMO_BOTTOM)
      if (lx < 0 || ly < 0 || lx > GIZMO_SIZE || ly > GIZMO_SIZE) return null
      if (Math.hypot(lx - GIZMO_SIZE / 2, ly - GIZMO_SIZE / 2) < 14) return null // centre dead zone
      syncGizmoCamera()
      let best: { dir: THREE.Vector3; d: number } | null = null
      for (const [, dir] of GIZMO_DIRS) {
        const p = dir.clone().multiplyScalar(1.3).project(gizmoCamera)
        const px = ((p.x + 1) / 2) * GIZMO_SIZE
        const py = ((1 - p.y) / 2) * GIZMO_SIZE
        const d = Math.hypot(px - lx, py - ly)
        if (!best || d < best.d) best = { dir, d }
      }
      return best && best.d <= 40 ? best.dir.clone() : null
    }

    // orbit-preserving snap: keep the distance, just change the direction
    const snapToAxis = (dir: THREE.Vector3) => {
      const distance = camera.position.distanceTo(controls.target)
      camera.position.copy(controls.target).addScaledVector(dir, distance)
      controls.update()
    }

    const onUp = (e: PointerEvent) => {
      if (gizmoDrag) {
        gizmoDrag = false
        controls.enabled = true
        el.style.cursor = 'grab'
        if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId)
        // a still click (no real drag) on an axis badge still snaps to it
        if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return
      }
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return // it was a drag
      lastX = e.clientX
      lastY = e.clientY
      const axis = gizmoHit(e.clientX, e.clientY)
      if (axis) {
        snapToAxis(axis)
        return
      }
      propsRef.current.onSelect(pick())
    }
    const onLeave = () => propsRef.current.onHover(null)
    const onDouble = (e: MouseEvent) => {
      if (gizmoHit(e.clientX, e.clientY)) return // don't pick points behind the cube
      lastX = e.clientX
      lastY = e.clientY
      const id = pick()
      if (id) {
        propsRef.current.onSelect(id)
        propsRef.current.onRequestFocus?.()
      }
    }

    el.addEventListener('pointermove', onMove)
    el.addEventListener('pointerdown', onDown)
    el.addEventListener('pointerup', onUp)
    el.addEventListener('pointerleave', onLeave)
    el.addEventListener('dblclick', onDouble)

    return () => {
      el.removeEventListener('pointermove', onMove)
      el.removeEventListener('pointerdown', onDown)
      el.removeEventListener('pointerup', onUp)
      el.removeEventListener('pointerleave', onLeave)
      el.removeEventListener('dblclick', onDouble)
      renderer.setAnimationLoop(null)
      state.resizeObserver.disconnect()
      clearGroup(pointsGroup)
      clearGroup(linesGroup)
      clearGroup(labelsGroup)
      clearGroup(grid) // orbital rings
      starField.geometry.dispose()
      starMat.dispose()
      coreGlow.material.dispose() // sprite has no geometry; its map is shared
      composer.dispose()
      ;(scene.background as THREE.Texture | null)?.dispose?.()
      controls.dispose()
      renderer.dispose()
      renderer.domElement.remove()
      gizmoPanel.remove()
      stateRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- camera framing --------------------------------------------------
  useEffect(() => {
    const st = stateRef.current
    if (!st) return
    st.grid.visible = props.dim === 3
    st.controls.enableRotate = props.dim === 3
    st.controls.mouseButtons = {
      LEFT: props.dim === 3 ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    }
    st.controls.touches = {
      ONE: props.dim === 3 ? THREE.TOUCH.ROTATE : THREE.TOUCH.PAN,
      TWO: THREE.TOUCH.DOLLY_PAN,
    }
    if (props.dim === 3) st.camera.position.set(19, 15, 27)
    else st.camera.position.set(0, 0, 34)
    st.controls.target.set(0, 0, 0)
    st.controls.update()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.dim, props.resetKey, props.coords])

  // ---- focus a grouping (selected vector + its nearest neighbours) -----
  useEffect(() => {
    if (!props.focusKey) return
    const st = stateRef.current
    if (!st) return
    const { coords, selectedId, neighborIds, dim } = propsRef.current
    if (!coords) return

    const ids = new Set<string>()
    if (selectedId) ids.add(selectedId)
    for (const id of neighborIds) ids.add(id)
    if (ids.size === 0) for (const id of Object.keys(coords)) ids.add(id) // nothing selected -> frame all

    const pts: THREE.Vector3[] = []
    for (const id of ids) {
      const c = coords[id]
      if (!c) continue
      pts.push(new THREE.Vector3(c[0] * WORLD_SCALE, c[1] * WORLD_SCALE, dim === 3 ? (c[2] ?? 0) * WORLD_SCALE : 0))
    }
    if (!pts.length) return

    const box = new THREE.Box3().setFromPoints(pts)
    const centre = box.getCenter(new THREE.Vector3())
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1.6)
    const fovRad = (st.camera.fov * Math.PI) / 180
    const distance = Math.max(3, (radius / Math.sin(fovRad / 2)) * 1.3)

    // keep the current viewing direction, just move closer
    const dir = st.camera.position.clone().sub(st.controls.target)
    if (dir.lengthSq() < 1e-9) dir.set(0, 0, 1)
    dir.normalize()

    st.controls.target.copy(centre)
    st.camera.position.copy(centre).addScaledVector(dir, distance)
    st.controls.update()
  }, [props.focusKey])

  // ---- rebuild geometry ------------------------------------------------
  useEffect(() => {
    const st = stateRef.current
    if (!st) return

    clearGroup(st.pointsGroup)
    clearGroup(st.linesGroup)
    clearGroup(st.labelsGroup)

    const { vectors, enabledIds, coords, selectedId, neighborIds, neighborRanks, showLines, showLabels, dim, distanceLookup } =
      propsRef.current
    if (!coords) return

    const enabled = vectors.filter((v) => enabledIds.has(v.id) && coords[v.id])
    const n = enabled.length
    if (n === 0) return

    const r = pointRadius(n)
    const hasSelection = selectedId !== null
    const toWorld = (id: string): THREE.Vector3 => {
      const c = coords[id]
      return new THREE.Vector3(c[0] * WORLD_SCALE, c[1] * WORLD_SCALE, dim === 3 ? (c[2] ?? 0) * WORLD_SCALE : 0)
    }

    const positions = new Map<string, THREE.Vector3>()
    for (const v of enabled) positions.set(v.id, toWorld(v.id))

    // ---- points + point labels ----------------------------------------
    const showAllPointLabels = showLabels && n <= MAX_POINT_LABELS
    for (const v of enabled) {
      const pos = positions.get(v.id)!
      const isSel = v.id === selectedId
      const isNb = neighborIds.has(v.id)
      const isHov = v.id === propsRef.current.hoverId
      const cluster = String(v.metadata?.cluster ?? v.metadata?.source ?? '')
      const color = isSel
        ? '#ffd166' // amber — matches the legend's "selected" bullet
        : isNb
          ? '#6ee7b7' // leaf-green — matches "nearest neighbours"
          // map the cluster's hue around the 6-slot palette (sextants) so
          // distinct clusters stay distinct — plain %6 collides too often
          : COSMOS[Math.floor(hashHue(cluster || v.id) / (360 / COSMOS.length)) % COSMOS.length]
      const opacity = hasSelection && !isSel && !isNb ? 0.28 : 1

      const scale = isSel ? 1.6 : isHov ? 1.25 : 1
      const mat = new THREE.SpriteMaterial({
        map: cosmicTexture(hashHue(v.id)), // one of 4 procedural cosmic bodies
        color,
        transparent: true,
        opacity,
        // depth is written so lines tuck neatly behind a body's aura
        depthWrite: true,
      })
      const body = new THREE.Sprite(mat)
      body.position.copy(pos)
      body.userData.id = v.id
      body.userData.baseRadius = r * scale // drives the screen-size envelope
      st.pointsGroup.add(body)

      if (showAllPointLabels || isSel || isNb || isHov) {
        const rank = neighborRanks.get(v.id)
        const text = rank !== undefined ? `${rank}. ${labelFor(v)}` : labelFor(v)
        const label = makeLabel(
          text,
          isSel ? '#201a05' : '#f2f6ff',
          isSel ? 'rgba(255,209,102,0.95)' : 'rgba(14,11,36,0.86)',
          isSel ? 15 : 13, // target on-screen height in px
          1.5, // never larger than 1.5 world units (stops labels swallowing the graph)
          isSel ? '#ffd166' : isNb ? '#6ee7b7' : '#8b9dff', // HUD accent
        )
        label.userData.anchor = pos.clone()
        label.userData.offsetK = 1.6 // floats above its point by 1.6x its height
        // declutter priority: selected first, then ranked neighbours, then the rest
        label.userData.priority = isSel ? 0 : rank !== undefined ? rank : isNb ? 4 : 10
        label.position.copy(pos).add(new THREE.Vector3(0, 1.5 * 1.6, 0))
        st.labelsGroup.add(label)
      }
    }

    // ---- connections ---------------------------------------------------
    const normal: number[] = []
    const highlighted: number[] = []
    const pairLabels: Array<{ a: string; b: string }> = []

    if (showLines) {
      for (let i = 0; i < enabled.length; i++) {
        for (let j = i + 1; j < enabled.length; j++) {
          const a = enabled[i].id
          const b = enabled[j].id
          const pa = positions.get(a)!
          const pb = positions.get(b)!
          const target = a === selectedId || b === selectedId ? highlighted : normal
          target.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z)
          if (showLabels) pairLabels.push({ a, b })
        }
      }

      if (normal.length) {
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.Float32BufferAttribute(normal, 3))
        st.linesGroup.add(
          new THREE.LineSegments(
            geo,
            new THREE.LineBasicMaterial({ color: '#4fbf96', transparent: true, opacity: 0.42 }),
          ),
        )
      }
      if (highlighted.length) {
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.Float32BufferAttribute(highlighted, 3))
        st.linesGroup.add(
          new THREE.LineSegments(
            geo,
            new THREE.LineBasicMaterial({ color: '#ffd166', transparent: true, opacity: 0.85 }),
          ),
        )
      }
    }

    // ---- distance labels ----------------------------------------------
    if (showLines && showLabels && pairLabels.length) {
      const incidentOnly = pairLabels.length > MAX_DISTANCE_LABELS
      const shown = incidentOnly
        ? pairLabels.filter((p) => p.a === selectedId || p.b === selectedId)
        : pairLabels
      for (const { a, b } of shown) {
        const d = distanceLookup.get(pairKey(a, b))
        if (d === undefined) continue
        const mid = positions.get(a)!.clone().add(positions.get(b)!).multiplyScalar(0.5)
        const label = makeLabel(fmt(d), '#cfe0ff', 'rgba(12,10,30,0.88)', 11, 1.1, '#59d6ff')
        label.userData.anchor = mid.clone()
        label.userData.offsetK = 0 // sits exactly on the line's midpoint
        label.userData.priority = 20 // point labels win the declutter pass
        label.position.copy(mid)
        st.labelsGroup.add(label)
      }
      propsRef.current.onHint?.(
        incidentOnly
          ? `Graph has ${pairLabels.length} pairs — distance labels are shown only on the selected vector's connections.`
          : null,
      )
    } else {
      propsRef.current.onHint?.(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    props.coords,
    props.enabledIds,
    props.dim,
    props.showLines,
    props.showLabels,
    props.selectedId,
    props.hoverId,
    props.neighborIds,
    props.neighborRanks,
    props.distanceLookup,
  ])

  return <div ref={containerRef} className="graph-container" />
}
