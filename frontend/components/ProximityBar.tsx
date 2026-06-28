import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Colors, Fonts } from '../constants/theme';

interface Props { level: number } // 0–1

export default function ProximityBar({ level }: Props) {
  return (
    <View>
      <View style={styles.row}>
        <View style={styles.labelRow}>
          <View style={styles.dot} />
          <Text style={styles.label}>PROXIMITY · CAUTION</Text>
        </View>
        <Text style={styles.sub}>KEEP YOUR DISTANCE</Text>
      </View>
      <View style={styles.track}>
        <LinearGradient
          colors={[Colors.safe, Colors.amber, Colors.danger]}
          locations={[0.42, 0.76, 1]}
          start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}
          style={StyleSheet.absoluteFill}
        />
        <View style={[styles.thumb, { left: `${level * 100}%` as any }]} />
      </View>
      <View style={styles.legend}>
        <Text style={styles.legendText}>SAFE · 25 M+</Text>
        <Text style={styles.legendText}>CAUTION</Text>
        <Text style={styles.legendText}>DANGER · &lt;10 M</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  labelRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  dot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: Colors.gold },
  label: { fontFamily: Fonts.mono, fontSize: 9.5, letterSpacing: 1.2, color: Colors.gold },
  sub: { fontFamily: Fonts.mono, fontSize: 9, letterSpacing: 0.8, color: 'rgba(243,236,222,0.6)' },
  track: { height: 5, borderRadius: 3, overflow: 'hidden', position: 'relative' },
  thumb: {
    position: 'absolute', top: -4, bottom: -4,
    width: 2, backgroundColor: Colors.cream,
    shadowColor: '#000', shadowOpacity: 0.55, shadowRadius: 6,
  },
  legend: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 6 },
  legendText: { fontFamily: Fonts.mono, fontSize: 8, letterSpacing: 0.8, color: 'rgba(243,236,222,0.5)' },
});
