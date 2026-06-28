import { View, Text, TouchableOpacity, StyleSheet, Platform } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { Colors, Fonts } from '../constants/theme';

export default function UnclearScreen() {
  const BackdropView = Platform.OS === 'android' && (Platform.Version as number) < 31
    ? ({ children, style }: any) => (
        <View style={[style, { backgroundColor: 'rgba(15,11,5,0.65)' }]}>{children}</View>
      )
    : ({ children, style }: any) => (
        <BlurView intensity={80} style={style}>{children}</BlurView>
      );

  return (
    <View style={styles.root}>
      {/* Blurred backdrop photo */}
      <View style={styles.bg}>
        <Text style={styles.bgEmoji}>🌿</Text>
      </View>
      <BackdropView style={StyleSheet.absoluteFill} />

      <LinearGradient
        colors={['rgba(15,11,5,0.5)', 'rgba(15,11,5,0.2)', 'rgba(15,11,5,0.86)']}
        locations={[0, 0.36, 1]}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={styles.overlay} edges={['top', 'bottom']}>
        {/* Status */}
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>7:51</Text>
          <Text style={styles.statusText}>5G · 83%</Text>
        </View>

        {/* Center shape indicator */}
        <View style={styles.center}>
          <View style={styles.shapeDash}>
            <Text style={styles.shapeLabel}>SHAPE{'\n'}DETECTED</Text>
          </View>
        </View>

        {/* Bottom content */}
        <View style={styles.bottom}>
          <View style={styles.confBadge}>
            <Text style={styles.confText}>31% · UNCLEAR</Text>
          </View>
          <Text style={styles.headline}>Hmm — a little blurry!</Text>
          <Text style={styles.body}>
            I can make out a shape low in the grass, but I can't say what it is yet. Let's get a cleaner look.
          </Text>

          <View style={styles.actions}>
            <TouchableOpacity style={styles.primaryBtn} onPress={() => router.replace('/capture')}>
              <Text style={styles.primaryBtnText}>Try a clearer shot</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryBtn} onPress={() => router.replace('/capture')}>
              <Text style={styles.secondaryBtnText}>It's moving — track it</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryBtn} onPress={() => router.push('/chat')}>
              <Text style={styles.secondaryBtnText}>Describe what you see</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root:    { flex: 1, backgroundColor: Colors.dark },
  bg:      { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  bgEmoji: { fontSize: 140, opacity: 0.25 },
  overlay: { flex: 1 },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 24, height: 44, alignItems: 'center' },
  statusText: { fontFamily: Fonts.mono, fontSize: 12, color: Colors.cream },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  shapeDash: {
    width: 120, height: 120, borderRadius: 60,
    borderWidth: 1, borderColor: 'rgba(243,236,222,0.45)',
    borderStyle: 'dashed',
    alignItems: 'center', justifyContent: 'center',
    opacity: 0.8,
  },
  shapeLabel: { fontFamily: Fonts.mono, fontSize: 10, letterSpacing: 1.2, color: Colors.cream, textAlign: 'center' },
  bottom: { padding: 24, paddingBottom: 30 },
  confBadge: { alignSelf: 'flex-start', borderWidth: 1, borderColor: 'rgba(243,236,222,0.4)', paddingHorizontal: 11, paddingVertical: 6, borderRadius: 2, marginBottom: 18 },
  confText: { fontFamily: Fonts.mono, fontSize: 10, letterSpacing: 1, color: Colors.cream },
  headline: { fontFamily: Fonts.displayBold, fontSize: 38, lineHeight: 39, color: Colors.cream, marginBottom: 10 },
  body:     { fontFamily: Fonts.body, fontSize: 16.5, color: 'rgba(243,236,222,0.78)', lineHeight: 25.6, marginBottom: 22 },
  actions:  { gap: 11 },
  primaryBtn: { backgroundColor: Colors.amber, borderRadius: 30, paddingVertical: 15, alignItems: 'center' },
  primaryBtnText: { fontFamily: Fonts.display, fontSize: 18, color: Colors.dark },
  secondaryBtn: { borderWidth: 1, borderColor: 'rgba(243,236,222,0.4)', borderRadius: 30, paddingVertical: 14, alignItems: 'center' },
  secondaryBtnText: { fontFamily: Fonts.display, fontSize: 17, color: Colors.cream },
});
