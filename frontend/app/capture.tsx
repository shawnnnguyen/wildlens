import { useState, useRef } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  Alert, ActivityIndicator, Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import ScanReticle from '../components/ScanReticle';
import ProximityBar from '../components/ProximityBar';
import { Colors, Fonts } from '../constants/theme';
import { useSession, mockIdentify } from '../store/session';

type Mode = 'IDENTIFY' | 'ASK' | 'JOURNAL';

export default function CaptureScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState<Mode>('IDENTIFY');
  const [identifying, setIdentifying] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const { setSession, guideName } = useSession();

  if (!permission) return <View style={styles.root} />;

  if (!permission.granted) {
    return (
      <View style={[styles.root, styles.permCenter]}>
        <Text style={styles.permTitle}>Camera access needed</Text>
        <Text style={styles.permBody}>WildLens needs your camera to identify wildlife in the field.</Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant Access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  async function handleShutter() {
    if (identifying) return;
    setIdentifying(true);
    await new Promise(r => setTimeout(r, 1200));
    const uri = 'mock://photo';
    const { scenario, subject } = mockIdentify(uri);
    setSession({
      photoUri: uri,
      species:     subject.species,
      taxon:       subject.taxon,
      traits:      subject.traits,
      confidence:  subject.confidence,
      isDangerous: subject.isDangerous,
      scenario,
    });
    setIdentifying(false);
    router.push(`/${scenario}`);
  }

  return (
    <View style={styles.root}>
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
      <LinearGradient
        colors={['rgba(15,11,5,0.6)', 'rgba(15,11,5,0.05)', 'rgba(15,11,5,0.1)', 'rgba(15,11,5,0.88)']}
        locations={[0, 0.26, 0.6, 1]}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={styles.overlay} edges={['top', 'bottom']}>
        {/* Status bar */}
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>7:42</Text>
          <Text style={styles.statusText}>5G · 86%</Text>
        </View>

        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.appName}>WILDLENS</Text>
            <Text style={styles.appSub}>AI FIELD GUIDE</Text>
          </View>
          <View style={styles.coords}>
            <Text style={styles.coordText}>−2.331° 34.821°</Text>
            <Text style={styles.coordText}>MAASAI MARA</Text>
          </View>
        </View>

        {/* Focus meta */}
        <View style={styles.focusMeta}>
          <Text style={styles.metaText}>EFL 26MM · ƒ/1.8</Text>
          <View style={styles.sharpRow}>
            <View style={styles.sharpDot} />
            <Text style={styles.metaText}>FOCUS 0.84 · SHARP</Text>
          </View>
        </View>

        {/* Reticle */}
        <View style={styles.reticleArea}>
          <ScanReticle />
          <View style={styles.reticleMeta}>
            <View style={styles.distPill}>
              <Text style={styles.distValue}>≈ 14 M</Text>
              <Text style={styles.distLabel}>EST. FROM FOCAL LENGTH</Text>
            </View>
            <Text style={styles.focusReady}>✓ IN FOCUS — TAP TO IDENTIFY</Text>
          </View>
        </View>

        {/* Proximity bar */}
        <View style={styles.proximityArea}>
          <ProximityBar level={0.64} />
        </View>

        {/* Mode tabs + controls */}
        <View style={styles.bottomArea}>
          <View style={styles.tabs}>
            {(['IDENTIFY', 'ASK', 'JOURNAL'] as Mode[]).map(m => (
              <TouchableOpacity key={m} onPress={() => setMode(m)}>
                <Text style={[styles.tab, mode === m && styles.tabActive]}>{m}</Text>
                {mode === m && <View style={styles.tabUnderline} />}
              </TouchableOpacity>
            ))}
          </View>
          <View style={styles.controls}>
            <View style={styles.galleryThumb} />
            <TouchableOpacity style={styles.shutter} onPress={handleShutter} disabled={identifying}>
              {identifying
                ? <ActivityIndicator color={Colors.dark} />
                : <View style={styles.shutterInner} />}
            </TouchableOpacity>
            <Text style={styles.flashLabel}>FLASH{'\n'}AUTO</Text>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.dark },
  permCenter: { alignItems: 'center', justifyContent: 'center', gap: 16, padding: 32 },
  permTitle: { fontFamily: Fonts.display, fontSize: 28, color: Colors.cream, textAlign: 'center' },
  permBody:  { fontFamily: Fonts.body, fontSize: 16, color: Colors.textDim, textAlign: 'center', lineHeight: 24 },
  permBtn:   { backgroundColor: Colors.amber, borderRadius: 30, paddingHorizontal: 28, paddingVertical: 14, marginTop: 8 },
  permBtnText: { fontFamily: Fonts.display, fontSize: 18, color: Colors.dark },
  overlay:   { flex: 1 },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 24, paddingVertical: 4 },
  statusText:{ fontFamily: Fonts.mono, fontSize: 12, color: Colors.cream },
  header:    { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', paddingHorizontal: 22, marginTop: 6 },
  appName:   { fontFamily: Fonts.display, fontWeight: '600', fontSize: 22, letterSpacing: 2.5, color: Colors.cream },
  appSub:    { fontFamily: Fonts.mono, fontSize: 9, letterSpacing: 2.6, color: 'rgba(243,236,222,0.6)', marginTop: 2 },
  coords:    { alignItems: 'flex-end' },
  coordText: { fontFamily: Fonts.mono, fontSize: 9.5, letterSpacing: 0.6, color: 'rgba(243,236,222,0.72)', lineHeight: 16 },
  focusMeta: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 22, marginTop: 12 },
  metaText:  { fontFamily: Fonts.mono, fontSize: 9, letterSpacing: 1, color: 'rgba(243,236,222,0.7)' },
  sharpRow:  { flexDirection: 'row', alignItems: 'center', gap: 6 },
  sharpDot:  { width: 6, height: 6, borderRadius: 3, backgroundColor: Colors.safe },
  reticleArea: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 22 },
  reticleMeta: { alignItems: 'center', gap: 9 },
  distPill: {
    flexDirection: 'row', alignItems: 'center', gap: 9,
    backgroundColor: 'rgba(15,11,5,0.45)',
    borderWidth: 1, borderColor: 'rgba(243,236,222,0.22)',
    paddingHorizontal: 13, paddingVertical: 7, borderRadius: 30,
  },
  distValue: { fontFamily: Fonts.mono, fontWeight: '700', fontSize: 11, color: Colors.cream },
  distLabel: { fontFamily: Fonts.mono, fontSize: 9, letterSpacing: 1, color: 'rgba(243,236,222,0.55)' },
  focusReady:{ fontFamily: Fonts.mono, fontSize: 10, letterSpacing: 2, color: 'rgba(243,236,222,0.82)' },
  proximityArea: { paddingHorizontal: 22, marginBottom: 14 },
  bottomArea: { paddingHorizontal: 24, paddingBottom: 8 },
  tabs: { flexDirection: 'row', justifyContent: 'center', gap: 30, marginBottom: 24 },
  tab: { fontFamily: Fonts.mono, fontSize: 10, letterSpacing: 1.8, color: 'rgba(243,236,222,0.5)', paddingBottom: 6 },
  tabActive: { color: Colors.amber },
  tabUnderline: { height: 2, backgroundColor: Colors.amber, marginTop: -2 },
  controls: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  galleryThumb: {
    width: 48, height: 48, borderRadius: 11,
    borderWidth: 1, borderColor: 'rgba(243,236,222,0.4)',
    backgroundColor: 'rgba(243,236,222,0.09)',
  },
  shutter: {
    width: 76, height: 76, borderRadius: 38,
    borderWidth: 3, borderColor: 'rgba(243,236,222,0.92)',
    alignItems: 'center', justifyContent: 'center',
  },
  shutterInner: { width: 60, height: 60, borderRadius: 30, backgroundColor: Colors.cream },
  flashLabel: { fontFamily: Fonts.mono, fontSize: 9, letterSpacing: 1, color: 'rgba(243,236,222,0.7)', textAlign: 'center', width: 48, lineHeight: 15 },
});
