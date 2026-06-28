import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import SafetyBanner from '../components/SafetyBanner';
import { Colors, Fonts } from '../constants/theme';
import { useSession } from '../store/session';

const STEPS = [
  'Roll up your windows and stay inside the vehicle.',
  'No sudden movements, no raised voices.',
  'Switch off your flash — never shoot with light.',
];

export default function SafetyScreen() {
  const { species, guideName } = useSession();

  return (
    <View style={styles.root}>
      {/* Background placeholder */}
      <View style={styles.bg}>
        <Text style={styles.bgEmoji}>🐘</Text>
      </View>

      <LinearGradient
        colors={['rgba(120,22,12,0.42)', 'rgba(30,10,6,0.25)', 'rgba(24,8,5,0.78)', 'rgba(20,6,4,0.97)']}
        locations={[0, 0.38, 0.7, 1]}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={styles.overlay} edges={['top', 'bottom']}>
        {/* Status */}
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>7:48</Text>
          <Text style={styles.statusText}>5G · 84%</Text>
        </View>

        {/* Alert banner */}
        <SafetyBanner label={`SAFETY ALERT · ${species.toUpperCase()} 12 M`} />

        <View style={{ flex: 1 }} />

        {/* Content */}
        <View style={styles.content}>
          <Text style={styles.headline}>Stay completely{'\n'}still.</Text>
          <Text style={styles.subhead}>
            This bull is close and alert. {guideName} flagged it the instant your photo came in.
          </Text>

          <View style={styles.steps}>
            {STEPS.map((step, i) => (
              <View key={i} style={styles.step}>
                <Text style={styles.stepNum}>0{i + 1}</Text>
                <Text style={styles.stepText}>{step}</Text>
              </View>
            ))}
          </View>

          <TouchableOpacity style={styles.confirmBtn} onPress={() => router.replace('/capture')}>
            <Text style={styles.confirmText}>I understand — keep me safe</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root:    { flex: 1, backgroundColor: Colors.dark },
  bg:      { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  bgEmoji: { fontSize: 120, opacity: 0.3 },
  overlay: { flex: 1 },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 24, height: 44, alignItems: 'center' },
  statusText: { fontFamily: Fonts.mono, fontSize: 12, color: Colors.cream },
  content: { padding: 24, paddingBottom: 30 },
  headline: { fontFamily: Fonts.displayBold, fontSize: 44, lineHeight: 43, color: '#fff', marginBottom: 10 },
  subhead:  { fontFamily: Fonts.body, fontSize: 16.5, color: 'rgba(255,255,255,0.82)', lineHeight: 24.75, marginBottom: 22 },
  steps:    { gap: 13, marginBottom: 24 },
  step:     { flexDirection: 'row', gap: 13, alignItems: 'flex-start' },
  stepNum:  { fontFamily: Fonts.mono, fontSize: 12, color: '#f0a094', marginTop: 3 },
  stepText: { fontFamily: Fonts.body, fontSize: 16.5, color: '#fff', lineHeight: 23.1, flex: 1 },
  confirmBtn: { backgroundColor: '#fff', borderRadius: 30, paddingVertical: 15, alignItems: 'center' },
  confirmText:{ fontFamily: Fonts.display, fontSize: 18, color: Colors.dark },
});
